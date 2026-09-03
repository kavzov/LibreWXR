// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Joshua Kimsey
// Modifications Copyright (C) 2026 Igor Kavzov
//! Optional single-threaded weather and radar rendering kernels for LibreWXR.

use numpy::{
    ndarray::{Array2, Array3},
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

struct Plan<'a> {
    r0: &'a [i32],
    r1: &'a [i32],
    c0: &'a [i32],
    c1: &'a [i32],
    dr: &'a [f32],
    dc: &'a [f32],
    valid: &'a [bool],
    rows: usize,
    cols: usize,
}

#[allow(clippy::too_many_arguments)]
fn plan_from_arrays<'a>(
    r0: &'a PyReadonlyArray2<'a, i32>,
    r1: &'a PyReadonlyArray2<'a, i32>,
    c0: &'a PyReadonlyArray2<'a, i32>,
    c1: &'a PyReadonlyArray2<'a, i32>,
    dr: &'a PyReadonlyArray2<'a, f32>,
    dc: &'a PyReadonlyArray2<'a, f32>,
    valid: &'a PyReadonlyArray2<'a, bool>,
) -> PyResult<Plan<'a>> {
    let shape = r0.shape();
    let expected = [shape[0], shape[1]];
    for (name, actual) in [
        ("r1", r1.shape()),
        ("c0", c0.shape()),
        ("c1", c1.shape()),
        ("dr", dr.shape()),
        ("dc", dc.shape()),
        ("valid", valid.shape()),
    ] {
        if actual != expected {
            return Err(PyValueError::new_err(format!(
                "sampling plan {name} shape {actual:?} does not match r0 {expected:?}"
            )));
        }
    }
    let contiguous =
        |name: &str| PyValueError::new_err(format!("sampling plan {name} must be C-contiguous"));
    Ok(Plan {
        r0: r0.as_slice().map_err(|_| contiguous("r0"))?,
        r1: r1.as_slice().map_err(|_| contiguous("r1"))?,
        c0: c0.as_slice().map_err(|_| contiguous("c0"))?,
        c1: c1.as_slice().map_err(|_| contiguous("c1"))?,
        dr: dr.as_slice().map_err(|_| contiguous("dr"))?,
        dc: dc.as_slice().map_err(|_| contiguous("dc"))?,
        valid: valid.as_slice().map_err(|_| contiguous("valid"))?,
        rows: expected[0],
        cols: expected[1],
    })
}

fn frame_slice<'a, T: numpy::Element>(
    frame: &'a PyReadonlyArray2<'a, T>,
    name: &str,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = frame.shape();
    let values = frame
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?;
    Ok((values, shape[0], shape[1]))
}

fn checked_index(row: i32, col: i32, height: usize, width: usize) -> Result<usize, String> {
    if row < 0 || col < 0 || row as usize >= height || col as usize >= width {
        return Err(format!(
            "sampling plan index ({row}, {col}) is outside frame shape ({height}, {width})"
        ));
    }
    Ok(row as usize * width + col as usize)
}

fn encoded_sample<T>(
    frame: &[T],
    height: usize,
    width: usize,
    plan: &Plan<'_>,
    index: usize,
    nodata: Option<T>,
    bilinear: bool,
) -> Result<Option<f32>, String>
where
    T: Copy + PartialEq + Into<f32>,
{
    let indexes = [
        checked_index(plan.r0[index], plan.c0[index], height, width)?,
        checked_index(plan.r0[index], plan.c1[index], height, width)?,
        checked_index(plan.r1[index], plan.c0[index], height, width)?,
        checked_index(plan.r1[index], plan.c1[index], height, width)?,
    ];
    let dr = plan.dr[index];
    let dc = plan.dc[index];
    if !dr.is_finite()
        || !dc.is_finite()
        || !(0.0..=1.0).contains(&dr)
        || !(0.0..=1.0).contains(&dc)
    {
        return Err(format!(
            "sampling weights at output index {index} must be finite and within [0, 1]"
        ));
    }
    if !plan.valid[index] {
        return Ok(None);
    }
    let missing = |value: T| nodata.is_some_and(|sentinel| value == sentinel);
    if !bilinear {
        let value = frame[indexes[0]];
        return Ok((!missing(value)).then(|| value.into()));
    }
    let weights = [
        (1.0 - dr) * (1.0 - dc),
        (1.0 - dr) * dc,
        dr * (1.0 - dc),
        dr * dc,
    ];
    let mut weighted = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for (source_index, weight) in indexes.into_iter().zip(weights) {
        let value = frame[source_index];
        if !missing(value) {
            weighted += value.into() * weight;
            weight_sum += weight;
        }
    }
    Ok((weight_sum > 0.0).then(|| weighted / weight_sum))
}

#[allow(clippy::too_many_arguments)]
fn sample_kernel<T>(
    frame: &[T],
    height: usize,
    width: usize,
    plan: &Plan<'_>,
    scale: f32,
    offset: f32,
    nodata: Option<T>,
    bilinear: bool,
) -> Result<Vec<f32>, String>
where
    T: Copy + PartialEq + Into<f32>,
{
    if !scale.is_finite() || !offset.is_finite() {
        return Err("scale and offset must be finite".to_string());
    }
    let mut output = Vec::with_capacity(plan.valid.len());
    for index in 0..plan.valid.len() {
        let sampled = encoded_sample(frame, height, width, plan, index, nodata, bilinear)?;
        output.push(sampled.map_or(f32::NAN, |value| value * scale + offset));
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn temporal_kernel<T>(
    frame_a: &[T],
    frame_b: &[T],
    height: usize,
    width: usize,
    plan: &Plan<'_>,
    alpha: f32,
    scale: f32,
    offset: f32,
    nodata: Option<T>,
    bilinear: bool,
) -> Result<Vec<f32>, String>
where
    T: Copy + PartialEq + Into<f32>,
{
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err("temporal alpha must be finite and within [0, 1]".to_string());
    }
    if !scale.is_finite() || !offset.is_finite() {
        return Err("scale and offset must be finite".to_string());
    }
    let mut output = Vec::with_capacity(plan.valid.len());
    for index in 0..plan.valid.len() {
        let a = encoded_sample(frame_a, height, width, plan, index, nodata, bilinear)?;
        let b = encoded_sample(frame_b, height, width, plan, index, nodata, bilinear)?;
        let encoded = match (a, b) {
            (Some(a), Some(b)) => Some(a + alpha * (b - a)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        output.push(encoded.map_or(f32::NAN, |value| value * scale + offset));
    }
    Ok(output)
}

macro_rules! sample_function {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (frame, r0, r1, c0, c1, dr, dc, valid, scale, offset, nodata=None, bilinear=true))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            frame: PyReadonlyArray2<'py, $ty>,
            r0: PyReadonlyArray2<'py, i32>,
            r1: PyReadonlyArray2<'py, i32>,
            c0: PyReadonlyArray2<'py, i32>,
            c1: PyReadonlyArray2<'py, i32>,
            dr: PyReadonlyArray2<'py, f32>,
            dc: PyReadonlyArray2<'py, f32>,
            valid: PyReadonlyArray2<'py, bool>,
            scale: f32,
            offset: f32,
            nodata: Option<$ty>,
            bilinear: bool,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let (frame, height, width) = frame_slice(&frame, "frame")?;
            let plan = plan_from_arrays(&r0, &r1, &c0, &c1, &dr, &dc, &valid)?;
            let result = py
                .allow_threads(|| sample_kernel(frame, height, width, &plan, scale, offset, nodata, bilinear))
                .map_err(PyIndexError::new_err)?;
            let array = Array2::from_shape_vec((plan.rows, plan.cols), result)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(array.into_pyarray(py))
        }
    };
}

macro_rules! temporal_function {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (frame_a, frame_b, r0, r1, c0, c1, dr, dc, valid, alpha, scale, offset, nodata=None, bilinear=true))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            frame_a: PyReadonlyArray2<'py, $ty>,
            frame_b: PyReadonlyArray2<'py, $ty>,
            r0: PyReadonlyArray2<'py, i32>,
            r1: PyReadonlyArray2<'py, i32>,
            c0: PyReadonlyArray2<'py, i32>,
            c1: PyReadonlyArray2<'py, i32>,
            dr: PyReadonlyArray2<'py, f32>,
            dc: PyReadonlyArray2<'py, f32>,
            valid: PyReadonlyArray2<'py, bool>,
            alpha: f32,
            scale: f32,
            offset: f32,
            nodata: Option<$ty>,
            bilinear: bool,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let (frame_a, height, width) = frame_slice(&frame_a, "frame_a")?;
            let (frame_b, height_b, width_b) = frame_slice(&frame_b, "frame_b")?;
            if (height, width) != (height_b, width_b) {
                return Err(PyValueError::new_err(format!(
                    "frame_b shape ({height_b}, {width_b}) does not match frame_a ({height}, {width})"
                )));
            }
            let plan = plan_from_arrays(&r0, &r1, &c0, &c1, &dr, &dc, &valid)?;
            let result = py
                .allow_threads(|| temporal_kernel(frame_a, frame_b, height, width, &plan, alpha, scale, offset, nodata, bilinear))
                .map_err(PyIndexError::new_err)?;
            let array = Array2::from_shape_vec((plan.rows, plan.cols), result)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(array.into_pyarray(py))
        }
    };
}

sample_function!(sample_i16, i16);
sample_function!(sample_u16, u16);
temporal_function!(sample_temporal_i16, i16);
temporal_function!(sample_temporal_u16, u16);

fn pair_shape<'a>(
    left: &'a PyReadonlyArray2<'a, f32>,
    right: &'a PyReadonlyArray2<'a, f32>,
    left_name: &str,
    right_name: &str,
) -> PyResult<(&'a [f32], &'a [f32], usize, usize)> {
    if left.shape() != right.shape() {
        return Err(PyValueError::new_err(format!(
            "{right_name} shape {:?} does not match {left_name} {:?}",
            right.shape(),
            left.shape()
        )));
    }
    let left_values = left
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{left_name} must be C-contiguous")))?;
    let right_values = right
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{right_name} must be C-contiguous")))?;
    Ok((left_values, right_values, left.shape()[0], left.shape()[1]))
}

#[pyfunction]
fn sample_derived_humidity<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray2<'py, f32>,
    dewpoint: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (temperature, dewpoint, rows, cols) =
        pair_shape(&temperature, &dewpoint, "temperature", "dewpoint")?;
    let result = py.allow_threads(|| {
        temperature
            .iter()
            .zip(dewpoint)
            .map(|(&temperature, &dewpoint)| {
                if !temperature.is_finite() || !dewpoint.is_finite() {
                    return f32::NAN;
                }
                let exponent = 17.625 * dewpoint / (243.04 + dewpoint)
                    - 17.625 * temperature / (243.04 + temperature);
                (100.0 * exponent.exp()).clamp(0.0, 100.0)
            })
            .collect::<Vec<_>>()
    });
    let array = Array2::from_shape_vec((rows, cols), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
fn sample_wind_speed<'py>(
    py: Python<'py>,
    wind_u: PyReadonlyArray2<'py, f32>,
    wind_v: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (wind_u, wind_v, rows, cols) = pair_shape(&wind_u, &wind_v, "wind_u", "wind_v")?;
    let result = py.allow_threads(|| {
        wind_u
            .iter()
            .zip(wind_v)
            .map(|(&u, &v)| {
                if u.is_finite() && v.is_finite() {
                    u.hypot(v)
                } else {
                    f32::NAN
                }
            })
            .collect::<Vec<_>>()
    });
    let array = Array2::from_shape_vec((rows, cols), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
fn sample_radar_bilinear_u8<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray2<'py, u8>,
    row: PyReadonlyArray2<'py, f32>,
    col: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if row.shape() != col.shape() {
        return Err(PyValueError::new_err(format!(
            "col shape {:?} does not match row {:?}",
            col.shape(),
            row.shape(),
        )));
    }
    let (frame, height, width) = frame_slice(&frame, "frame")?;
    let rows = row.shape()[0];
    let cols = row.shape()[1];
    let row = row
        .as_slice()
        .map_err(|_| PyValueError::new_err("row must be C-contiguous"))?;
    let col = col
        .as_slice()
        .map_err(|_| PyValueError::new_err("col must be C-contiguous"))?;
    let result = py
        .allow_threads(|| {
            let mut output = Vec::with_capacity(row.len());
            for (&row_f, &col_f) in row.iter().zip(col) {
                if !row_f.is_finite() || !col_f.is_finite() {
                    return Err(format!(
                        "radar sample coordinate ({row_f}, {col_f}) must be finite"
                    ));
                }
                // Masked coordinate plans encode pixels outside the source
                // grid as -1 in both planes.  Emit a transparent radar value
                // directly instead of requiring a second full-size integer
                // index pair solely to zero those pixels after sampling.
                if row_f < 0.0
                    || col_f < 0.0
                    || row_f > (height - 1) as f32
                    || col_f > (width - 1) as f32
                {
                    output.push(0);
                    continue;
                }
                let r0 = row_f.floor() as usize;
                let c0 = col_f.floor() as usize;
                let r1 = (r0 + 1).min(height - 1);
                let c1 = (c0 + 1).min(width - 1);
                let v00 = frame[r0 * width + c0];
                let v01 = frame[r0 * width + c1];
                let v10 = frame[r1 * width + c0];
                let v11 = frame[r1 * width + c1];
                if v00 == 0 || v01 == 0 || v10 == 0 || v11 == 0 {
                    output.push(v00);
                    continue;
                }
                let dr = row_f - r0 as f32;
                let dc = col_f - c0 as f32;
                let interpolated = v00 as f32 * (1.0 - dr) * (1.0 - dc)
                    + v01 as f32 * (1.0 - dr) * dc
                    + v10 as f32 * dr * (1.0 - dc)
                    + v11 as f32 * dr * dc;
                output.push((interpolated + 0.5).clamp(0.0, 255.0) as u8);
            }
            Ok(output)
        })
        .map_err(PyIndexError::new_err)?;
    let array = Array2::from_shape_vec((rows, cols), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (radar, model, model_raw, feather, blend_weight, pixel_threshold=None))]
fn blend_radar_nowcast_u8<'py>(
    py: Python<'py>,
    radar: PyReadonlyArray2<'py, u8>,
    model: PyReadonlyArray2<'py, f32>,
    model_raw: PyReadonlyArray2<'py, u8>,
    feather: PyReadonlyArray2<'py, f32>,
    blend_weight: f32,
    pixel_threshold: Option<u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = radar.shape();
    for (name, actual) in [
        ("model", model.shape()),
        ("model_raw", model_raw.shape()),
        ("feather", feather.shape()),
    ] {
        if actual != shape {
            return Err(PyValueError::new_err(format!(
                "{name} shape {actual:?} does not match radar {shape:?}"
            )));
        }
    }
    if !blend_weight.is_finite() || !(0.0..=1.0).contains(&blend_weight) {
        return Err(PyValueError::new_err(
            "blend_weight must be finite and within [0, 1]",
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let radar = radar
        .as_slice()
        .map_err(|_| PyValueError::new_err("radar must be C-contiguous"))?;
    let model = model
        .as_slice()
        .map_err(|_| PyValueError::new_err("model must be C-contiguous"))?;
    let model_raw = model_raw
        .as_slice()
        .map_err(|_| PyValueError::new_err("model_raw must be C-contiguous"))?;
    let feather = feather
        .as_slice()
        .map_err(|_| PyValueError::new_err("feather must be C-contiguous"))?;
    let result = py.allow_threads(|| {
        radar
            .iter()
            .zip(model)
            .zip(model_raw)
            .zip(feather)
            .map(|(((&radar, &model), &model_raw), &feather)| {
                if radar == 0 && model_raw == 0 {
                    return 0;
                }
                let mut model = model;
                if blend_weight > 0.0
                    && pixel_threshold
                        .is_some_and(|threshold| model < threshold as f32 && radar >= threshold)
                {
                    model = pixel_threshold.expect("threshold checked") as f32;
                }
                let weight = blend_weight * feather.clamp(0.0, 1.0);
                (weight * radar as f32 + (1.0 - weight) * model + 0.5).clamp(0.0, 255.0) as u8
            })
            .collect::<Vec<_>>()
    });
    let array = Array2::from_shape_vec((rows, cols), result)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (values, rain_lut, snow_lut=None, snow_mask=None, display_threshold=None))]
fn colorize_radar_u8<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, u8>,
    rain_lut: PyReadonlyArray2<'py, u8>,
    snow_lut: Option<PyReadonlyArray2<'py, u8>>,
    snow_mask: Option<PyReadonlyArray2<'py, bool>>,
    display_threshold: Option<u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    if rain_lut.shape() != [256, 4] {
        return Err(PyValueError::new_err("rain_lut must have shape (256, 4)"));
    }
    if snow_lut.as_ref().is_some_and(|lut| lut.shape() != [256, 4]) {
        return Err(PyValueError::new_err("snow_lut must have shape (256, 4)"));
    }
    if snow_mask
        .as_ref()
        .is_some_and(|mask| mask.shape() != values.shape())
    {
        return Err(PyValueError::new_err(format!(
            "snow_mask shape {:?} does not match values {:?}",
            snow_mask
                .as_ref()
                .expect("shape mismatch requires mask")
                .shape(),
            values.shape(),
        )));
    }
    if snow_mask.is_some() != snow_lut.is_some() {
        return Err(PyValueError::new_err(
            "snow_lut and snow_mask must either both be provided or both omitted",
        ));
    }
    let rows = values.shape()[0];
    let cols = values.shape()[1];
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
    let rain_lut = rain_lut
        .as_slice()
        .map_err(|_| PyValueError::new_err("rain_lut must be C-contiguous"))?;
    let snow_lut = snow_lut
        .as_ref()
        .map(|lut| {
            lut.as_slice()
                .map_err(|_| PyValueError::new_err("snow_lut must be C-contiguous"))
        })
        .transpose()?;
    let snow_mask = snow_mask
        .as_ref()
        .map(|mask| {
            mask.as_slice()
                .map_err(|_| PyValueError::new_err("snow_mask must be C-contiguous"))
        })
        .transpose()?;
    let output = py.allow_threads(|| {
        let mut rgba = Vec::with_capacity(values.len() * 4);
        for (index, &raw_value) in values.iter().enumerate() {
            let value = if display_threshold.is_some_and(|threshold| raw_value < threshold) {
                0
            } else {
                raw_value
            };
            let lut = if snow_mask.is_some_and(|mask| mask[index]) {
                snow_lut.expect("snow LUT validated with mask")
            } else {
                rain_lut
            };
            rgba.extend_from_slice(&lut[value as usize * 4..value as usize * 4 + 4]);
        }
        rgba
    });
    let array = Array3::from_shape_vec((rows, cols, 4), output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

fn encode_png_bytes(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, png::EncodingError> {
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        encoder.set_filter(png::FilterType::Paeth);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
    }
    Ok(encoded)
}

#[pyfunction]
fn encode_png_rgba<'py>(
    py: Python<'py>,
    rgba: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = rgba.shape();
    if shape.len() != 3 || shape[2] != 4 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err(
            "rgba must have non-empty shape (height, width, 4)",
        ));
    }
    let height = u32::try_from(shape[0])
        .map_err(|_| PyValueError::new_err("rgba height exceeds PNG limits"))?;
    let width = u32::try_from(shape[1])
        .map_err(|_| PyValueError::new_err("rgba width exceeds PNG limits"))?;
    let rgba = rgba
        .as_slice()
        .map_err(|_| PyValueError::new_err("rgba must be C-contiguous"))?;
    let encoded = py
        .allow_threads(|| encode_png_bytes(rgba, width, height))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &encoded))
}

#[pymodule]
fn _librewxr_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(sample_i16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_u16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_temporal_i16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_temporal_u16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_derived_humidity, module)?)?;
    module.add_function(wrap_pyfunction!(sample_wind_speed, module)?)?;
    module.add_function(wrap_pyfunction!(sample_radar_bilinear_u8, module)?)?;
    module.add_function(wrap_pyfunction!(blend_radar_nowcast_u8, module)?)?;
    module.add_function(wrap_pyfunction!(colorize_radar_u8, module)?)?;
    module.add_function(wrap_pyfunction!(encode_png_rgba, module)?)?;
    Ok(())
}

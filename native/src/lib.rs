// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Joshua Kimsey
//! Optional single-threaded weather sampling kernels for LibreWXR.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

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

#[pymodule]
fn _librewxr_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(sample_i16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_u16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_temporal_i16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_temporal_u16, module)?)?;
    module.add_function(wrap_pyfunction!(sample_derived_humidity, module)?)?;
    module.add_function(wrap_pyfunction!(sample_wind_speed, module)?)?;
    Ok(())
}

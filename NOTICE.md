# LibreWXR fork notice

This repository is a modified fork of
[JoshuaKimsey/LibreWXR](https://github.com/JoshuaKimsey/LibreWXR).

- Original LibreWXR: Copyright (C) 2026 Joshua Kimsey.
- Fork-specific development began on 2026-08-06.
- Fork modifications: Copyright (C) 2026 Igor Kavzov and the other
  contributors identified in the Git history.
- Maintained corresponding source for this fork:
  <https://github.com/kavzov/LibreWXR>.

Material fork-specific work includes global scalar weather fields, radar point
nowcasts and motion data, motion-compensated animation, multi-worker deployment
hardening, atomic shared state and cache generations, shared coordinate and
encoded-tile stores, render-stage instrumentation, and native Rust sampling,
compositing, colourisation, and PNG encoding kernels. The Git history is the
authoritative record of individual changes and their dates.

LibreWXR and these modifications are distributed under the GNU Affero General
Public License, version 3 or (at your option) any later version. See
[LICENSE](LICENSE). Existing upstream copyright, attribution, licence, and
warranty notices are retained.

The upstream maintainer advertises separate commercial licensing for rights
they control. Those terms do not automatically grant rights to fork-specific
modifications owned by other contributors.

The HTML examples under `examples/` are separately available under the MIT
license in `examples/LICENSE-examples`. Data products consumed or rendered by
LibreWXR remain subject to their respective source licences and attribution
requirements documented in the README and source packages.

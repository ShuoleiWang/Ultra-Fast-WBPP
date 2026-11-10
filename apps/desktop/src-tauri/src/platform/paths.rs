//! The spelling of a path that the engine, a solver and the interface share.

use std::path::{Path, PathBuf};

/// Longest plain Windows path in UTF-16 units: `MAX_PATH` less its terminator.
const LEGACY_PATH_UNITS: usize = 259;

/// `path` in the plain Windows spelling when that names the same file.
///
/// `canonicalize` returns Windows paths in the verbatim form (`\\?\C:\…`,
/// `\\?\UNC\server\share\…`); the interface and the engine's own inventory
/// hold the plain one (`C:\…`, `\\server\share\…`), and the two never compare
/// equal.  The verbatim form is kept when only it can name the file: a path
/// over the legacy length limit, another kind of verbatim path, or a
/// component the plain form would reinterpret.  Paths of other platforms are
/// returned as they are.
pub(crate) fn plain_path(path: &Path) -> PathBuf {
    path.to_str()
        .and_then(plain_windows_path)
        .map_or_else(|| path.to_path_buf(), PathBuf::from)
}

fn plain_windows_path(path: &str) -> Option<String> {
    let rest = path.strip_prefix(r"\\?\")?;
    let (plain, names, required) = match rest.strip_prefix(r"UNC\") {
        // A server and a share name.
        Some(share) => (format!(r"\\{share}"), share, 2),
        None => {
            let drive = rest.as_bytes();
            if drive.len() < 3 || !drive[0].is_ascii_alphabetic() || &drive[1..3] != br":\" {
                return None;
            }
            (rest.to_owned(), &rest[3..], 0)
        }
    };
    let mut names: Vec<&str> = names.split('\\').collect();
    // A root keeps its separator (`C:\`); `C:` alone is another path.
    if names.last() == Some(&"") {
        names.pop();
    }
    let ordinary = names.len() >= required && names.iter().all(|name| ordinary_name(name));
    (ordinary && plain.encode_utf16().count() <= LEGACY_PATH_UNITS).then_some(plain)
}

/// Whether the plain form reads `name` as the verbatim form does: Win32
/// normalisation drops a trailing dot or space, resolves `.` and `..`, splits
/// at `/` and maps the DOS device names (with any extension) to devices.
fn ordinary_name(name: &str) -> bool {
    // The part before the first dot decides, whatever spaces follow it.
    let stem = name.split('.').next().unwrap_or(name).trim_end_matches(' ');
    let device = ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
        .iter()
        .any(|device| stem.eq_ignore_ascii_case(device))
        || ["COM", "LPT"].iter().any(|port| {
            let mut number = stem.get(3..).unwrap_or("").chars();
            stem.get(..3)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(port))
                && matches!(
                    (number.next(), number.next()),
                    (Some(digit), None) if digit.is_ascii_digit() || "¹²³".contains(digit)
                )
        });
    !(name.is_empty()
        || device
        || name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|c| c.is_control() || r#"<>:"/|?*"#.contains(c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(path: &str) -> String {
        plain_path(Path::new(path)).to_string_lossy().into_owned()
    }

    #[test]
    fn verbatim_drive_and_share_paths_take_the_plain_spelling() {
        for (verbatim, expected) in [
            (
                r"\\?\C:\Users\example\天文\M31",
                r"C:\Users\example\天文\M31",
            ),
            (
                r"\\?\d:\Astro\NGC 7331\L_0001.fits",
                r"d:\Astro\NGC 7331\L_0001.fits",
            ),
            (r"\\?\D:\", r"D:\"),
            (r"\\?\UNC\server\share\x", r"\\server\share\x"),
            (
                r"\\?\UNC\nas.local\天文\M31\L.fits",
                r"\\nas.local\天文\M31\L.fits",
            ),
            (r"\\?\UNC\server\share", r"\\server\share"),
            (r"\\?\UNC\server\share\", r"\\server\share\"),
            // Names that only resemble a device or carry dots inside.
            (
                r"\\?\C:\data\CONSOLE\COM10\nul2.fits",
                r"C:\data\CONSOLE\COM10\nul2.fits",
            ),
            (r"\\?\C:\data\.hidden\a.b.c", r"C:\data\.hidden\a.b.c"),
        ] {
            assert_eq!(plain(verbatim), expected);
        }
    }

    #[test]
    fn paths_only_the_verbatim_form_can_name_keep_it() {
        // 300 characters in the plain spelling.
        let long = format!(r"\\?\C:\{0}\{0}\{1}", "a".repeat(99), "b".repeat(97));
        assert_eq!(long.chars().count(), 300 + r"\\?\".len());
        for kept in [
            long.as_str(),
            // A trailing dot or space, which the plain form drops.
            r"\\?\C:\data\night.",
            r"\\?\C:\data\night \L.fits",
            // DOS device names, with or without an extension, in any case.
            r"\\?\C:\data\NUL\x.fits",
            r"\\?\C:\data\con.txt",
            r"\\?\C:\data\Aux",
            r"\\?\C:\data\COM1",
            r"\\?\C:\data\lpt9.fits",
            r"\\?\C:\data\COM².fits",
            r"\\?\C:\data\CONIN$",
            // Components the plain form would resolve or split.
            r"\\?\C:\data\..\other",
            r"\\?\C:\data\.\x",
            r"\\?\C:\data\\x",
            r"\\?\C:\data/x",
            r"\\?\C:\data\a:b",
            r"\\?\C:\data\what?.fits",
            // Not a drive or a share.
            r"\\?\C:",
            r"\\?\C:relative",
            r"\\?\Volume{b75e2c83-0000-0000-0000-602f00000000}\x",
            r"\\?\GLOBALROOT\Device\HarddiskVolume3\x",
            r"\\?\UNC\server",
            r"\\?\UNC\\share\x",
            r"\\?\UNC\server\\x",
        ] {
            assert_eq!(plain(kept), kept);
        }
    }

    #[test]
    fn the_length_limit_counts_utf16_units_of_the_plain_form() {
        let fits = format!(r"\\?\C:\{}", "a".repeat(256));
        assert_eq!(plain(&fits).encode_utf16().count(), 259);
        assert!(!plain(&fits).starts_with(r"\\?\"));
        let over = format!(r"\\?\C:\{}", "a".repeat(257));
        assert_eq!(plain(&over), over);
        // One character outside the BMP is two units.
        let astral = format!(r"\\?\C:\{}🔭", "a".repeat(255));
        assert_eq!(plain(&astral), astral);
        let share = format!(r"\\?\UNC\server\share\{}", "a".repeat(244));
        assert_eq!(plain(&share).encode_utf16().count(), 259);
        let share_over = format!(r"\\?\UNC\server\share\{}", "a".repeat(245));
        assert_eq!(plain(&share_over), share_over);
    }

    #[test]
    fn plain_and_foreign_paths_are_returned_unchanged() {
        for unchanged in [
            r"C:\Users\example\天文\M31",
            r"\\server\share\x",
            r"\\.\C:\x",
            "/Users/example/Astro/M31",
            "/Volumes/Astro 2/盾牌座/L_0001.fits",
            "relative/path",
            "",
        ] {
            assert_eq!(plain(unchanged), unchanged);
        }
    }
}

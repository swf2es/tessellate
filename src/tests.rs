use super::*;
use std::f64::consts::PI;
use std::vec;
use std::vec::Vec;

/// The contours joined, with their ends.
fn join(contours: &[Vec<[f64; 2]>]) -> (Vec<f64>, Vec<u32>) {
    let mut coords = Vec::new();
    let mut ends = Vec::new();
    for contour in contours {
        for p in contour {
            coords.extend_from_slice(p);
        }

        ends.push((coords.len() / 2) as u32);
    }

    (coords, ends)
}

fn area(t: &Triangles) -> f64 {
    let p = |i: u32| {
        let i = i as usize * 2;
        (t.positions[i] as f64, t.positions[i + 1] as f64)
    };
    t.indices
        .chunks_exact(3)
        .map(|k| {
            let (a, b, c) = (p(k[0]), p(k[1]), p(k[2]));
            ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() / 2.0
        })
        .sum()
}

fn winding(contours: &[Vec<[f64; 2]>], x: f64, y: f64) -> i32 {
    let mut w = 0;
    for contour in contours {
        for i in 0..contour.len() {
            let a = contour[i];
            let b = contour[(i + 1) % contour.len()];
            let side = (b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1]);
            if a[1] <= y && b[1] > y && side > 0.0 {
                w += 1;
            } else if a[1] > y && b[1] <= y && side < 0.0 {
                w -= 1;
            }
        }
    }

    w
}

fn near_edge(contours: &[Vec<[f64; 2]>], x: f64, y: f64, margin: f64) -> bool {
    contours.iter().any(|contour| {
        (0..contour.len()).any(|i| {
            let a = contour[i];
            let b = contour[(i + 1) % contour.len()];
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length = dx * dx + dy * dy;
            let t = if length == 0.0 {
                0.0
            } else {
                (((x - a[0]) * dx + (y - a[1]) * dy) / length).clamp(0.0, 1.0)
            };
            let (px, py) = (a[0] + t * dx - x, a[1] + t * dy - y);
            px * px + py * py < margin * margin
        })
    })
}

fn covers(t: &Triangles, x: f64, y: f64) -> usize {
    let p = |i: u32| {
        let i = i as usize * 2;
        (t.positions[i] as f64, t.positions[i + 1] as f64)
    };
    t.indices
        .chunks_exact(3)
        .filter(|k| {
            let (a, b, c) = (p(k[0]), p(k[1]), p(k[2]));
            let d1 = (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0);
            let d2 = (c.0 - b.0) * (y - b.1) - (c.1 - b.1) * (x - b.0);
            let d3 = (a.0 - c.0) * (y - c.1) - (a.1 - c.1) * (x - c.0);
            (d1 > 0.0 && d2 > 0.0 && d3 > 0.0) || (d1 < 0.0 && d2 < 0.0 && d3 < 0.0)
        })
        .count()
}

/// Every sample of a grid over the contours, but those within `margin` of an edge, is covered
/// once where the rule fills it and not at all where it does not.
fn check_coverage(contours: &[Vec<[f64; 2]>], nonzero: bool, t: &Triangles, steps: usize) {
    let points = contours.iter().flatten();
    let x0 = points.clone().map(|p| p[0]).fold(f64::MAX, f64::min);
    let x1 = points.clone().map(|p| p[0]).fold(f64::MIN, f64::max);
    let y0 = points.clone().map(|p| p[1]).fold(f64::MAX, f64::min);
    let y1 = points.map(|p| p[1]).fold(f64::MIN, f64::max);
    let margin = (x1 - x0).max(y1 - y0) * 1e-5;
    for i in 0..steps {
        for j in 0..steps {
            // Off the grid's lines, where axis-aligned edges lie.
            let x = x0 + (x1 - x0) * (i as f64 + 0.371) / steps as f64;
            let y = y0 + (y1 - y0) * (j as f64 + 0.587) / steps as f64;
            if near_edge(contours, x, y, margin) {
                continue;
            }

            let w = winding(contours, x, y);
            let inside = if nonzero { w != 0 } else { w % 2 != 0 };
            assert_eq!(
                covers(t, x, y),
                inside as usize,
                "at ({x}, {y}), winding {w}"
            );
        }
    }
}

fn tessellate(contours: &[Vec<[f64; 2]>], nonzero: bool) -> Triangles {
    let (coords, ends) = join(contours);
    fill(&coords, &ends, nonzero).expect("filled")
}

fn square(x: f64, y: f64, side: f64) -> Vec<[f64; 2]> {
    vec![[x, y], [x + side, y], [x + side, y + side], [x, y + side]]
}

fn star() -> Vec<[f64; 2]> {
    (0..5)
        .map(|k| {
            let a = PI / 2.0 + k as f64 * 4.0 * PI / 5.0;
            [100.0 * a.cos(), 100.0 * a.sin()]
        })
        .collect()
}

/// `count` wobbly ellipses of `points` each, every one turned a little from the last.
fn deck(count: usize, points: usize) -> Vec<Vec<[f64; 2]>> {
    (0..count)
        .map(|c| {
            let a = c as f64 * 0.004;
            (0..points)
                .map(|k| {
                    let u = k as f64 / points as f64 * PI * 2.0;
                    let r = 200.0 + 12.0 * (u * 9.0).sin();
                    let (x, y) = (r * u.cos() * 0.7, r * u.sin());
                    [
                        300.0 + x * a.cos() - y * a.sin(),
                        300.0 + x * a.sin() + y * a.cos(),
                    ]
                })
                .collect()
        })
        .collect()
}

#[test]
fn fills_a_square() {
    let contours = [square(0.0, 0.0, 10.0)];
    for nonzero in [false, true] {
        let t = tessellate(&contours, nonzero);
        assert_eq!(t.indices.len(), 6);
        assert_eq!(area(&t), 100.0);
        check_coverage(&contours, nonzero, &t, 16);
    }
}

#[test]
fn leaves_a_hole() {
    // The hole wound against the outline, so that both rules leave it.
    let mut hole = square(2.0, 2.0, 6.0);
    hole.reverse();
    let contours = [square(0.0, 0.0, 10.0), hole];
    for nonzero in [false, true] {
        let t = tessellate(&contours, nonzero);
        assert!((area(&t) - 64.0).abs() < 1e-9);
        check_coverage(&contours, nonzero, &t, 32);
    }

    // Wound the same way, the non-zero rule fills it.
    let contours = [square(0.0, 0.0, 10.0), square(2.0, 2.0, 6.0)];
    assert!((area(&tessellate(&contours, true)) - 100.0).abs() < 1e-9);
    assert!((area(&tessellate(&contours, false)) - 64.0).abs() < 1e-9);
}

#[test]
fn fills_a_crossing_star_by_either_rule() {
    let contours = [star()];
    // The pentagram's five points, and the pentagon they leave between them.
    let outer = 100.0;
    let inner = outer * (2.0 * PI / 5.0).cos() / (PI / 5.0).cos();
    let pentagon = 2.5 * inner * inner * (2.0 * PI / 5.0).sin();
    let points = 5.0 * outer * inner * (PI / 5.0).sin() - pentagon;
    let evenodd = tessellate(&contours, false);
    let nonzero = tessellate(&contours, true);
    assert!((area(&evenodd) - points).abs() < 1e-3 * points);
    assert!((area(&nonzero) - (points + pentagon)).abs() < 1e-3 * points);
    check_coverage(&contours, false, &evenodd, 64);
    check_coverage(&contours, true, &nonzero, 64);
}

#[test]
fn fills_a_crossing_deck_by_either_rule() {
    let contours = deck(20, 200);
    for nonzero in [false, true] {
        let t = tessellate(&contours, nonzero);
        check_coverage(&contours, nonzero, &t, 96);
    }
}

#[test]
fn fills_a_dense_deck_in_good_time() {
    let contours = deck(95, 200);
    let start = std::time::Instant::now();
    let t = tessellate(&contours, false);
    // libtess2 takes seconds over it in wasm, lyon a tenth of one; natively, less.
    assert!(start.elapsed().as_secs_f64() < 1.0, "{:?}", start.elapsed());
    check_coverage(&contours, false, &t, 48);
}

#[test]
fn refuses_what_is_not_finite_or_in_range() {
    let good = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e7, -838_861.0] {
        let mut coords = good;
        coords[3] = bad;
        assert_eq!(fill(&coords, &[3], false).unwrap_err(), REFUSED, "{bad}");
    }

    // The largest coordinate that fits in twips is taken.
    let mut coords = good;
    coords[2] = 838_860.0;
    assert!(fill(&coords, &[3], false).is_ok());
}

#[test]
fn refuses_ends_that_do_not_close_the_points() {
    let coords = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0];
    for ends in [&[][..], &[3], &[5], &[2, 2, 4], &[3, 2], &[0, 4]] {
        assert_eq!(fill(&coords, ends, false).unwrap_err(), REFUSED, "{ends:?}");
    }

    assert_eq!(fill(&coords[..7], &[3], false).unwrap_err(), REFUSED);
    assert!(fill(&coords, &[4], false).is_ok());
    assert!(fill(&coords, &[1, 4], false).is_ok());
}

#[test]
fn the_c_function_checks_its_arguments() {
    let coords = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0];
    let ends = [3u32];
    let mut positions: *const f32 = core::ptr::null();
    let mut vertices = 0;
    let mut indices: *const u32 = core::ptr::null();
    let mut index_count = 0;
    let call = |count, contours, nonzero, p: &mut _, v: &mut _, i: &mut _, n: &mut _| unsafe {
        lyon_fill(
            coords.as_ptr(),
            count,
            ends.as_ptr(),
            contours,
            nonzero,
            p,
            v,
            i,
            n,
        )
    };
    for (count, contours, nonzero) in [(3, 1, 2), (3, 1, -1), (2, 3, 0), (MOST_POINTS + 1, 1, 0)] {
        let status = call(
            count,
            contours,
            nonzero,
            &mut positions,
            &mut vertices,
            &mut indices,
            &mut index_count,
        );
        assert_eq!(status, REFUSED);
        assert!(positions.is_null());
    }

    let status = call(
        3,
        1,
        1,
        &mut positions,
        &mut vertices,
        &mut indices,
        &mut index_count,
    );
    assert_eq!(status, OK);
    assert_eq!((vertices, index_count), (3, 3));
    let positions = unsafe { core::slice::from_raw_parts(positions, vertices * 2) };
    let indices = unsafe { core::slice::from_raw_parts(indices, index_count) };
    let mut sorted = indices.to_vec();
    sorted.sort();
    assert_eq!(sorted, [0, 1, 2]);
    let mut points: Vec<[f32; 2]> = positions.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(points, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]);
}

/// A sliver a sixtieth of a pixel wide: lyon's absolute epsilons lose it filled in pixels,
/// which is why the crate fills in twips.
#[test]
fn keeps_a_sub_pixel_sliver() {
    let contours = [vec![[0.0, 0.0], [40.0, 0.0], [40.0, 1.0 / 60.0]]];
    let (coords, ends) = join(&contours);
    let expected = 40.0 / 60.0 / 2.0;
    let twips = fill(&coords, &ends, false).unwrap();
    assert!((area(&twips) - expected).abs() < 1e-3 * expected);
    let pixels = fill_at(&coords, &ends, false, 1.0).unwrap();
    assert!(area(&pixels) < expected / 2.0, "{}", area(&pixels));
}

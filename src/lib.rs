//! lyon's fill tessellator behind one C function, for swf2es's render WebAssembly.
//!
//! `lyon_fill` takes flattened contours, as f64 pairs with each contour ending at its point in
//! `ends`, and a fill rule, and leaves f32 positions and u32 triangle indices in memory from the
//! allocator. The library is `no_std` and allocates through the C `malloc`, `realloc` and `free`
//! of whatever it is linked into: in swf2es, the render module's bump arena, which the host resets
//! before each call, so nothing here is ever freed one allocation at a time.
//!
//! Failures are returned as status codes. A panic inside lyon, which this crate's own checks are
//! there to prevent, traps on wasm32 (`unreachable`), and so does an allocation lyon makes that
//! fails: only the crate's own buffers report running out (`NO_MEMORY`). The host then discards
//! the instance.
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::vec::Vec;
use lyon_tessellation::math::{Point, point};
use lyon_tessellation::path::PathEvent;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, FillVertexConstructor,
    VertexBuffers,
};

/// The fill is tessellated in twentieths of the input's unit, twips where the input is pixels,
/// about the centre of its box: lyon's epsilons are absolute, and at pixel scale they drop
/// sub-pixel slivers, while far from the origin f32 keeps too few of a coordinate's bits. A twip
/// is a whole number in f32 up to `LIMIT`.
const SCALE: f64 = 20.0;
/// The largest coordinate, scaled about the centre, that f32 holds every whole number up to.
const LIMIT: f64 = 16_777_216.0;
/// The largest coordinate taken at all, so that every position given back is a finite f32.
const FARTHEST: f64 = 1e30;
/// The most points one call takes, which keeps every size below in range.
pub const MOST_POINTS: usize = 1 << 24;

/// Filled: the outputs are set.
pub const OK: i32 = 0;
/// lyon could not fill the contours.
pub const FAILED: i32 = -1;
/// The input was refused: a coordinate not finite, past ±1e30 or more than 838,860.8 from the
/// centre of the points' box, or `ends` not closing
/// contours of at least one point each that use every point.
pub const REFUSED: i32 = -2;
/// The allocator ran out of memory for the crate's own buffers. lyon's own allocations trap
/// instead.
pub const NO_MEMORY: i32 = -3;

#[cfg(not(test))]
mod allocator {
    use core::alloc::{GlobalAlloc, Layout};
    use core::ffi::c_void;
    use core::ptr::null_mut;

    // The host's: in swf2es, packages/native/src/render/arena.zig's, 16-byte aligned.
    unsafe extern "C" {
        fn malloc(size: usize) -> *mut c_void;
        fn realloc(pointer: *mut c_void, size: usize) -> *mut c_void;
        fn free(pointer: *mut c_void);
    }

    const ALIGNMENT: usize = 16;

    struct Host;

    unsafe impl GlobalAlloc for Host {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.align() > ALIGNMENT {
                return null_mut();
            }

            unsafe { malloc(layout.size()) as *mut u8 }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, _: Layout) {
            unsafe { free(pointer as *mut c_void) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if layout.align() > ALIGNMENT {
                return null_mut();
            }

            unsafe { realloc(pointer as *mut c_void, size) as *mut u8 }
        }
    }

    #[global_allocator]
    static HOST: Host = Host;
}

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// Elsewhere the library is only ever checked or tested, but a staticlib needs a handler.
#[cfg(all(not(target_arch = "wasm32"), not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe extern "C" {
        fn abort() -> !;
    }

    unsafe { abort() }
}

/// The contours as lyon's path events, each closed, read from the points in place.
struct Events<'a> {
    points: &'a [Point],
    ends: &'a [u32],
    contour: usize,
    at: usize,
    start: usize,
}

impl Iterator for Events<'_> {
    type Item = PathEvent;

    fn next(&mut self) -> Option<PathEvent> {
        let &end = self.ends.get(self.contour)?;
        let end = end as usize;
        if self.at == self.start {
            self.at += 1;
            return Some(PathEvent::Begin {
                at: self.points[self.start],
            });
        }

        if self.at < end {
            let event = PathEvent::Line {
                from: self.points[self.at - 1],
                to: self.points[self.at],
            };
            self.at += 1;
            return Some(event);
        }

        let event = PathEvent::End {
            last: self.points[end - 1],
            first: self.points[self.start],
            close: true,
        };
        self.contour += 1;
        self.start = end;
        self.at = end;
        Some(event)
    }
}

/// Positions back in the input's unit and place, in f64 before they narrow to f32.
struct Unscaled {
    scale: f64,
    x: f64,
    y: f64,
}

impl FillVertexConstructor<[f32; 2]> for Unscaled {
    fn new_vertex(&mut self, vertex: FillVertex) -> [f32; 2] {
        let p = vertex.position();
        [
            (p.x as f64 / self.scale + self.x) as f32,
            (p.y as f64 / self.scale + self.y) as f32,
        ]
    }
}

/// A fill's triangles: two f32s a vertex, three indices a triangle.
#[derive(Debug, Default)]
pub struct Triangles {
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Whether `ends` closes contours of a point or more over all of `count` points.
fn closes(ends: &[u32], count: usize) -> bool {
    let mut start = 0;
    for &end in ends {
        let end = end as usize;
        if end <= start || end > count {
            return false;
        }

        start = end;
    }

    start == count
}

/// Fills the contours of `coords`, x and y a point, each ending at its point in `ends`, under the
/// non-zero rule or else the even-odd one. Errors are the status codes.
pub fn fill(coords: &[f64], ends: &[u32], nonzero: bool) -> Result<Triangles, i32> {
    fill_at(coords, ends, nonzero, SCALE, true)
}

fn fill_at(
    coords: &[f64],
    ends: &[u32],
    nonzero: bool,
    scale: f64,
    centred: bool,
) -> Result<Triangles, i32> {
    let count = coords.len() / 2;
    if coords.len() % 2 != 0 || count > MOST_POINTS || !closes(ends, count) {
        return Err(REFUSED);
    }

    let (mut left, mut right) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut top, mut bottom) = (f64::INFINITY, f64::NEG_INFINITY);
    for pair in coords.chunks_exact(2) {
        // NaN fails the comparison.
        if !(pair[0].abs() <= FARTHEST && pair[1].abs() <= FARTHEST) {
            return Err(REFUSED);
        }

        left = left.min(pair[0]);
        right = right.max(pair[0]);
        top = top.min(pair[1]);
        bottom = bottom.max(pair[1]);
    }

    let (cx, cy) = if centred && count > 0 {
        ((left + right) / 2.0, (top + bottom) / 2.0)
    } else {
        (0.0, 0.0)
    };
    let mut points: Vec<Point> = Vec::new();
    points.try_reserve_exact(count).map_err(|_| NO_MEMORY)?;
    for pair in coords.chunks_exact(2) {
        let x = (pair[0] - cx) * scale;
        let y = (pair[1] - cy) * scale;
        if !(x.abs() <= LIMIT && y.abs() <= LIMIT) {
            return Err(REFUSED);
        }

        points.push(point(x as f32, y as f32));
    }

    // Room for the crossings' new vertices; a fill with more grows the buffers.
    let mut output: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    output
        .vertices
        .try_reserve(count + count / 4 + 16)
        .map_err(|_| NO_MEMORY)?;
    output
        .indices
        .try_reserve(count * 3 + 48)
        .map_err(|_| NO_MEMORY)?;

    let events = Events {
        points: &points,
        ends,
        contour: 0,
        at: 0,
        start: 0,
    };
    let rule = if nonzero {
        FillRule::NonZero
    } else {
        FillRule::EvenOdd
    };
    // The tolerance flattens curves, and the contours have none.
    let options = FillOptions::tolerance(0.02).with_fill_rule(rule);
    let mut builder = BuffersBuilder::new(
        &mut output,
        Unscaled {
            scale,
            x: cx,
            y: cy,
        },
    );
    if FillTessellator::new()
        .tessellate(events, &options, &mut builder)
        .is_err()
    {
        return Err(FAILED);
    }

    let vertices = output.vertices;
    let (pointer, length, capacity) = {
        let mut vertices = core::mem::ManuallyDrop::new(vertices);
        (vertices.as_mut_ptr(), vertices.len(), vertices.capacity())
    };
    // [f32; 2] is two f32s in a row, its alignment f32's.
    let positions = unsafe { Vec::from_raw_parts(pointer as *mut f32, length * 2, capacity * 2) };
    Ok(Triangles {
        positions,
        indices: output.indices,
    })
}

/// Fills `count` points at `coords`, x and y each, as contours ending at the `contours` indices
/// at `ends`, under the non-zero rule where `nonzero` is 1 and the even-odd one where 0.
///
/// Returns `OK` with `*positions` pointing at `*vertices` pairs of f32s and `*indices` at
/// `*index_count` u32s, three a triangle, all from the allocator and never freed here; or
/// `FAILED`, `REFUSED` or `NO_MEMORY`, the outputs untouched.
///
/// # Safety
///
/// `coords` must point at `count * 2` f64s and `ends` at `contours` u32s, each readable (either
/// may dangle where its count is 0), and each output pointer must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lyon_fill(
    coords: *const f64,
    count: usize,
    ends: *const u32,
    contours: usize,
    nonzero: i32,
    positions: *mut *const f32,
    vertices: *mut usize,
    indices: *mut *const u32,
    index_count: *mut usize,
) -> i32 {
    if count > MOST_POINTS || contours > count || !(nonzero == 0 || nonzero == 1) {
        return REFUSED;
    }

    if count > 0 && (coords.is_null() || ends.is_null()) {
        return REFUSED;
    }

    let (coords, ends) = if count == 0 {
        (&[][..], &[][..])
    } else {
        unsafe {
            (
                core::slice::from_raw_parts(coords, count * 2),
                core::slice::from_raw_parts(ends, contours),
            )
        }
    };
    match fill(coords, ends, nonzero == 1) {
        Ok(triangles) => {
            let triangles = core::mem::ManuallyDrop::new(triangles);
            unsafe {
                *positions = triangles.positions.as_ptr();
                *vertices = triangles.positions.len() / 2;
                *indices = triangles.indices.as_ptr();
                *index_count = triangles.indices.len();
            }
            OK
        }
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests;

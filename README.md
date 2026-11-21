# swf2es/tessellate

[lyon](https://github.com/nical/lyon)'s fill tessellator as a `no_std` static
library for `wasm32-unknown-unknown`, with one C function. It is linked into
the render WebAssembly module of [swf2es](https://github.com/swf2es/swf2es),
which fills with it the shapes libtess2 is slow at: many contours that cross
one another densely, as a morph of near-identical outlines does.

## The API

```c
int32_t lyon_fill(const double *coords, size_t count,
                  const uint32_t *ends, size_t contours,
                  int32_t nonzero,
                  const float **positions, size_t *vertices,
                  const uint32_t **indices, size_t *index_count);
```

- `coords` holds `count` points, x then y, as f64s. They are closed contours,
  each ending at its point in `ends` (the last end is `count`), each of one
  point or more; every contour is closed back to its first point.
- `nonzero` is 1 for the non-zero rule and 0 for the even-odd one.
- On success it returns 0, with `*positions` pointing at `*vertices` pairs of
  f32s and `*indices` at `*index_count` u32s, three a triangle.
- Otherwise it returns -1 where lyon could not fill the contours, -2 where it
  refused the input (a coordinate not finite or past ±838,860.8, `ends` that
  do not close the points, a rule other than 0 or 1), or -3 where memory for
  the output ran out; the outputs are left alone.

The points are filled in twentieths of their unit, twips where they are
pixels, and the positions given back in the unit again: lyon's epsilons are
absolute, and at pixel scale they drop sub-pixel slivers. A whole number of
twips is exact in f32 up to 2^24, which is where the range comes from.

The library imports nothing but `malloc`, `realloc` and `free`, through which
Rust's global allocator goes, and the stack pointer; the module it is linked
into provides them (in swf2es, a bump arena reset before each call, so the
outputs stay until then and nothing is freed one at a time). Allocations are
16-byte aligned at most. Errors are returned, not panicked; a panic, or an
allocation lyon makes that fails, executes `unreachable`, which traps, and
swf2es then discards the instance and fills the shape another way.

## How swf2es uses it

swf2es carries no Rust. Each release here attaches `libswf2es_tessellate.a`,
built by this repository's CI from the tag with the pinned toolchain, and its
SHA-256. swf2es commits that archive with its source and digest beside its
Zig render sources (`packages/native`), and its `build.zig` links it into the
render module with `addObjectFile`. A fill goes to lyon only where a count of
its contours' crossings says libtess2 would be slow; every other fill stays
with libtess2, which fills a few self-crossing even-odd shapes correctly that
lyon covers twice in places.

## Building

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.97.1 with the
wasm32 target and `llvm-tools`), and the dependencies in `Cargo.lock`, where
lyon_tessellation is pinned at 1.0.23 with its default features off.

```sh
cargo test --locked
cargo build --locked --release --target wasm32-unknown-unknown
```

The tests run on the host. The release build's archive holds the crate's
object, with lyon, core and alloc in it by LTO, and compiler_builtins, which
the object does not call; CI keeps the object alone in the archive it
publishes, after checking that the object imports only what is listed above
(see `.github/workflows/ci.yml`).

## Licence

Apache-2.0 (`LICENSE`). The notices of lyon, the crates it uses and Rust's
libraries are in `LICENSE.third-party`.

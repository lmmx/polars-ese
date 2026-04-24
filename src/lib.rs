use polars::prelude::*;
use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

fn embed_text_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Array(Box::new(DataType::Float32), ese::DIMENSIONS),
    ))
}

#[polars_expr(output_type_func=embed_text_output_type)]
pub fn embed_text(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let dim = ese::DIMENSIONS;
    let n = ca.len();

    // Collect texts; null -> "" (we'll set validity after).
    let texts: Vec<&str> = ca.into_iter().map(|o| o.unwrap_or("")).collect();

    // One batched call; rayon kicks in at >= 16.
    let vecs: Vec<[f32; ese::DIMENSIONS]> = ese::encode(texts);

    // Build as a List<f32> first, then cast to Array<f32, DIM>.
    // This mirrors what polars-luxical does and sidesteps any arrow
    // FixedSizeListArray builder API drift across polars versions.
    let mut builder =
        ListPrimitiveChunkedBuilder::<Float32Type>::new(ca.name().clone(), n, n * dim, DataType::Float32);

    for (i, v) in vecs.iter().enumerate() {
        if unsafe { ca.get_unchecked(i) }.is_some() {
            builder.append_slice(v);
        } else {
            builder.append_null();
        }
    }

    let list_series = builder.finish().into_series();
    list_series.cast(&DataType::Array(Box::new(DataType::Float32), dim))
}

#[pyfunction]
fn dimensions() -> usize {
    ese::DIMENSIONS
}

#[pymodule]
fn _polars_ese(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dimensions, m)?)?;
    m.add("DIMENSIONS", ese::DIMENSIONS)?;
    Ok(())
}
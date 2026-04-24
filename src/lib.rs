use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
use serde::Deserialize;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

#[derive(Deserialize)]
pub struct EmbedTextKwargs {
    // Reserved for future use; currently ESE has one model.
    #[allow(dead_code)]
    pub _placeholder: Option<String>,
}

fn embed_text_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Array(Box::new(DataType::Float32), ese::DIMENSIONS),
    ))
}

#[polars_expr(output_type_func=embed_text_output_type)]
pub fn embed_text(inputs: &[Series], _kwargs: EmbedTextKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let n = ca.len();
    let dim = ese::DIMENSIONS;

    // Collect non-null texts with their original row index.
    // ESE has no concept of "null input"; we embed "" for nulls and rely on
    // the validity bitmap to mask them out in the Array output.
    let texts: Vec<&str> = ca.into_iter().map(|o| o.unwrap_or("")).collect();

    // Single batched call — lets rayon parallelise when len >= 16.
    let vecs: Vec<[f32; ese::DIMENSIONS]> = ese::encode(texts);

    // Flatten into a contiguous buffer for the Array[f32, DIM] output.
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for v in vecs {
        flat.extend_from_slice(&v);
    }

    // Build validity from the input's null mask so null text -> null array row.
    let validity = ca.rechunk().chunks()[0]
        .as_any()
        .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
        .and_then(|a| a.validity().cloned());

    let values = PrimitiveArray::from_vec(flat);
    let arr = polars_arrow::array::FixedSizeListArray::new(
        polars_arrow::datatypes::ArrowDataType::FixedSizeList(
            Box::new(polars_arrow::datatypes::Field::new(
                "item".into(),
                polars_arrow::datatypes::ArrowDataType::Float32,
                true,
            )),
            dim,
        ),
        n,
        values.boxed(),
        validity,
    );

    Ok(unsafe {
        Series::_try_from_arrow_unchecked_with_md(
            ca.name().clone(),
            vec![Box::new(arr)],
            &polars_arrow::datatypes::ArrowDataType::FixedSizeList(
                Box::new(polars_arrow::datatypes::Field::new(
                    "item".into(),
                    polars_arrow::datatypes::ArrowDataType::Float32,
                    true,
                )),
                dim,
            ),
            None,
        )?
    })
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
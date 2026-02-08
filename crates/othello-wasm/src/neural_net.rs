use crate::model::demo::Model as DemoModel;
use burn::Tensor;
use burn::backend::{NdArray, WebGpu};
use burn::prelude::Backend;
use burn::tensor::{DataError, Shape, Transaction};
use othello::othello_game::{Color, OthelloGame};

/// An element in the policy vector in form ((row, col), probability)
pub(crate) type PolicyElement = ((usize, usize), f32);

/// A tuple containing the policy vector, and an evaluation scalar
pub(crate) type PolicyVectorWithEvaluation = (Vec<PolicyElement>, f32);

#[allow(clippy::large_enum_variant)]
/// The model is loaded to a specific backend
pub enum ModelType {
    /// The model is loaded to the NdArray backend
    WithNdArrayBackend(NeuralNet<NdArray<f32>>),

    /// The model is loaded to the WebGpu backend
    WithWgpuBackend(NeuralNet<WebGpu<f32, i32>>),
}

/// Neural network
pub struct NeuralNet<B: Backend> {
    model: DemoModel<B>,
}

impl<B: Backend> NeuralNet<B> {
    pub fn new(device: &B::Device) -> Self {
        tracing::info!("Initialising model");
        Self {
            model: DemoModel::from_embedded(device),
        }
    }

    pub async fn forward(&self, input: Tensor<B, 4>) -> Result<(Vec<f32>, f32), DataError> {
        let outputs = self.model.forward(input);

        let policy: Vec<f32> = outputs
            .0
            .to_data_async()
            .await
            .unwrap()
            .convert::<f32>()
            .to_vec()?;
        let value_array: Vec<f32> = outputs
            .1
            .to_data_async()
            .await
            .unwrap()
            .convert::<f32>()
            .to_vec()?;
        let value = value_array[0]; // Access the first element of the [1, 1] shape

        Ok((policy, value))
    }
}

fn encode_game<B: Backend>(game: &OthelloGame, player: Color) -> Tensor<B, 4> {
    let planes = game.encode(player);

    let mut data = Vec::with_capacity(1 * 2 * 8 * 8);

    for p in 0..2 {
        for r in 0..8 {
            for c in 0..8 {
                data.push(planes[p][r][c] as f32);
            }
        }
    }

    Tensor::<B, 4>::from_floats(data.as_slice(), &B::Device::default())
}

pub fn nn_evaluate<B: Backend>(
    model: &NeuralNet<B>,
    game: &OthelloGame,
    player: &Color,
) -> Result<(Vec<(usize, f32)>, f32), DataError> {
    let input = encode_game::<B>(game, *player);

    let (policy_tensor, value_tensor) = model.model.forward(input);

    // SINGLE synchronisation point
    let [policy_data, value_data] = Transaction::default()
        .register(policy_tensor)
        .register(value_tensor)
        .execute()
        .try_into()
        .unwrap();

    let policy: Vec<f32> = policy_data.convert::<f32>().to_vec()?;

    let value = value_data.convert::<f32>().as_slice()?[0];

    let sparse_policy = policy.into_iter().enumerate().collect();

    Ok((sparse_policy, value))
}

pub fn evaluate_test<B: Backend>(
    model: NeuralNet<B>,
    game: &OthelloGame,
    player: Color,
) -> Result<PolicyVectorWithEvaluation, DataError> {
    let mut data = vec![0.0f32; 1 * 2 * 8 * 8];

    let idx = |b, c, r, col| (((b * 2 + c) * 8) as usize + r) * 8 + col;

    for row in 0..8 {
        for col in 0..8 {
            if let Some(c) = game.get(row, col) {
                match (player, c) {
                    (Color::White, Color::White) | (Color::Black, Color::Black) => {
                        data[idx(0, 0, row, col)] = 1.0;
                    }
                    _ => {
                        data[idx(0, 1, row, col)] = 1.0;
                    }
                }
            }
        }
    }

    let input = Tensor::<B, 4>::from_floats(data.as_slice(), &B::Device::default());

    let outputs = model.forward(input);

    todo!()
}

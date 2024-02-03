import torch

from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def calculate_translation(
    image_alignment: ImageAlignment,
    input: torch.Tensor,
    reference_image: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    tvec = torch.zeros((input.shape[0], 2))

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input),
        batch_size=batch_size,
        shuffle=False,
    )
    start_position: int = 0
    for input_batch in data_loader:
        assert len(input_batch) == 1

        end_position = start_position + input_batch[0].shape[0]

        tvec_temp = image_alignment.dry_run_translation(
            input=input_batch[0],
            new_reference_image=reference_image,
        )

        assert tvec_temp is not None

        tvec[start_position:end_position, :] = tvec_temp

        start_position += input_batch[0].shape[0]

    return tvec

import torch

from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def calculate_rotation(
    image_alignment: ImageAlignment,
    input: torch.Tensor,
    reference_image: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    angle = torch.zeros((input.shape[0]))

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input),
        batch_size=batch_size,
        shuffle=False,
    )
    start_position: int = 0
    for input_batch in data_loader:
        assert len(input_batch) == 1

        end_position = start_position + input_batch[0].shape[0]

        angle_temp = image_alignment.dry_run_angle(
            input=input_batch[0],
            new_reference_image=reference_image,
        )

        assert angle_temp is not None

        angle[start_position:end_position] = angle_temp

        start_position += input_batch[0].shape[0]

    angle = torch.where(angle >= 180, 360.0 - angle, angle)
    angle = torch.where(angle <= -180, 360.0 + angle, angle)

    return angle

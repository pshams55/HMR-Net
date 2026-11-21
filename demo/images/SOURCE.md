# Demo images

The images in this folder are unmodified images from the **test split** of NWPU VHR-10
(split produced by `tools/convert_nwpu.py --test-ratio 0.3 --with-negatives --seed 0`), so they
were not seen during training. Together they cover all 10 NWPU classes.

| File | Classes in the ground truth |
|---|---|
| nwpu_609.jpg | basketball court, ground track field, tennis court |
| nwpu_305.jpg | ship, storage tank |
| nwpu_529.jpg | baseball diamond, bridge |
| nwpu_532.jpg | airplane, ship |
| nwpu_277.jpg | harbor, ship |
| nwpu_448.jpg | vehicle |

**Source:** NWPU VHR-10, obtained from the Hugging Face mirror `torchgeo/vhr10`
(redistributed without modification from the original release). License: CC BY-NC 4.0.

G. Cheng, P. Zhou, and J. Han, "Learning rotation-invariant convolutional neural networks for
object detection in VHR optical remote sensing images," IEEE Trans. Geosci. Remote Sens.,
vol. 54, no. 12, pp. 7405–7415, 2016.

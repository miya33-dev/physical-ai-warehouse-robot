# Third-party software and model notices

Project-authored source code in this repository is provided under AGPL-3.0-only; see LICENSE. Third-party packages, upstream model weights, and their bundled components retain their original licenses. This file does not relicense them.

## Ultralytics and YOLO26n
- Package: Ultralytics 8.4.198, https://github.com/ultralytics/ultralytics
- Upstream package license: AGPL-3.0, preserved verbatim in third_party/Ultralytics-LICENSE.txt.
- Base detector: YOLO26n, obtained from https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- Model documentation and licensing: https://docs.ultralytics.com/models/yolo26 and https://www.ultralytics.com/license
- This project fine-tunes the model for red_cube / blue_cube using original synthetic images. The upstream model is not presented as our original architecture or pretrained work.
- Model provenance, hashes, training configuration, limitations and evaluation are recorded in docs/YOLOモデルカード.md and configs/detector_model.json after training.
- Training scripts, data generator, evaluation scripts and configuration are included in source. Weights are stored separately because of binary size. When distributing trained weights, provide the matching source revision, this license/notice, model card, and reproduction instructions together; do not distribute an isolated checkpoint without its accompanying source information.

## Main runtime dependencies
- PyTorch 2.15.0: BSD-style and bundled third-party licenses; see the upstream distribution LICENSE and bundled notices. https://github.com/pytorch/pytorch
- Torchvision 0.29.2: BSD-style. https://github.com/pytorch/vision
- MuJoCo 3.3.7: Apache-2.0. https://github.com/google-deepmind/mujoco
- NumPy 2.5.4: BSD-3-Clause plus licenses of bundled components. https://numpy.org/doc/stable/license.html
- Pillow 13.0.0: MIT-CMU. https://github.com/python-pillow/Pillow
- ImageIO 2.37.0 and imageio-ffmpeg 0.6.0 Python package: BSD-2-Clause. https://github.com/imageio/imageio and https://github.com/imageio/imageio-ffmpeg
- OpenCV Python 5.0.0.93: Apache-2.0 and bundled component notices. https://github.com/opencv/opencv-python

The FFmpeg executable used by the installed video runtime has its own build-dependent licenses; the Python wrapper license does not replace those terms. The executable is not bundled in this source repository. The same applies to the locally downloaded Ollama runtime and its included components.

The language model Qwen3-4B-Instruct-2507 is a separate upstream model under Apache-2.0, not a project-authored model. Its model card is https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . The local quantized Ollama artifact retains its upstream terms. It is not part of the detector checkpoint.

A versioned dependency metadata inventory is in third_party/dependency_licenses.json. Original installed package distributions include their complete notices; the short descriptions above are not substitutes for those files.

## Data and assets
The training images are generated from this project's procedural MJCF scene and primitives. No lecture PDFs, course-only images, external photographs, private messages, account data or third-party visual assets are included. Source code to regenerate the images and annotations is provided. Simulator object IDs are used only to build labels and score evaluation, never as detector inference input.

## Source distribution scope
The source repository excludes local environments, downloaded runtimes, model binaries, synthetic dataset files, generated videos/PDFs, credentials and course handouts. Reproduction scripts download/install upstream dependencies separately. Exclusion from Git is a size/privacy choice, not an exemption from license obligations if those components are later distributed in another form.

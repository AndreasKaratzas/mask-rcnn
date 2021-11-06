# Custom Mask R-CNN implementation

* TODO: Introduction to project
* TODO: Delete this repository and uplaod everything from a new repository
* TODO: Save from `Visual`, don't display
* TODO: Check color list to be equal in element number to that of the masks

### Prerequisites

The only prerequisite is [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in order to clone the repository. The project was tested using a Python 3.8.3 interpreter.

### Installation

1. Open a terminal window
2. Upgrade `pip` using `python -m pip install --upgrade pip`
3. Clone repository using `git clone https://github.com/AndreasKaratzas/mask-rcnn.git`
4. Navigate to project directory with `cd mask-rcnn`
5. Create a virtual environment using `python -m venv ./venv`
6. Activate virtual environment with `./venv/Scripts/activate`
7. Install requirements using `pip install -r requirements.txt`
8. (Optional) To utilize your CUDA compatible GPU, use `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

### Usage

To train Mask R-CNN with custom data, use the training script:
```powershell
python train.py 
```

You can also test your models after training them using the testing script:
```powershell
python test.py
```

### Experiments

* TODO: Experiments stats and timing.

### System info

All tests were performed using a laptop: 
* Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
* Installed RAM: 16.0 GB (15.85 GB usable)
* Graphics card: NVIDIA GeForce GTX 1660 Ti

# Custom Mask R-CNN implementation

* TODO: Introduction to project

### Prerequisites

The only prerequisite is [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in order to clone the repository. The project was tested using Python 3.8.3 and 3.9.7 interpreters, on a Windows 10 OS.

### Installation

1. Open a terminal window
2. Clone repository using `git clone https://github.com/AndreasKaratzas/mask-rcnn.git`
3. Navigate to project directory with `cd mask-rcnn`
4. Create a virtual environment using `python -m venv ./venv`
5. Upgrade `pip` using `python -m pip install --upgrade pip`
6. Activate virtual environment with `./venv/Scripts/activate`
7. Install requirements using `pip install -r requirements.txt`
8. (Optional) To utilize your CUDA compatible GPU, use `pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`

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

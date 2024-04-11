# OpenEQA Dataset Viewer

## Quick Start

**Step 0:** Make sure you are in the viewer directory:

```bash
cd viewer
```

**Step 1:** Install [Flask](https://flask.palletsprojects.com/en/3.0.x/installation/#install-flask):

```bash
pip install Flask
```

**Step 2:** Get HM3D episode histories

You can download the episode histories for the HM3D split of OpenEQA from [here](https://www.dropbox.com/scl/fi/myw8et6xq83w22a1ktg4t/open-eqa-hm3d-videos-v0.tgz?rlkey=m0f7ri4r0bc77axgc8v2b74fr).

```bash
wget -O open-eqa-hm3d-videos-v0.tgz "https://www.dropbox.com/scl/fi/myw8et6xq83w22a1ktg4t/open-eqa-hm3d-videos-v0.tgz?rlkey=m0f7ri4r0bc77axgc8v2b74fr"
md5sum open-eqa-hm3d-videos-v0.tgz  # dc166d807e64dacdac791153c5d1c853
mkdir -p static/videos
tar -xzf open-eqa-hm3d-videos-v0.tgz -C static/videos
rm open-eqa-hm3d-videos-v0.tgz
```

Afterwards, your top-level directory should look like this:

```text
|- assets
|- ...
|- viewer
   |- static
      |- videos
         |- hm3d-v0
            |- 000-hm3d-BFRyYbPCCPE-0.mp4
            |- ...
   |- templates
   |- README.md
   |- app.py
| - ...
```

**Step 3:** Start the viewer app:

```bash
python app.py
```

Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Additional options

The quick start instructions above allow viewing question-answer pairs from the HM3D split of OpenEQA. To view samples from the ScanNet split, follow the instructions below.

### ScanNet Setup

**Step 1:** Download ScanNet data following the instructions [here](../data#ScanNet). Note: only the RGB frames from ScanNet are used to generate episode history videos in step 2.

**Step 2:** Convert the RGB frames into mp4 files.

```bash
python data/frames2videos.py --split scannet-v0
```

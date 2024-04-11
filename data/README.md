# OpenEQA Dataset

The OpenEQA dataset includes questions, answers, and episode histories $(Q,A^{\star},H)$.

The question-answer pairs are in [open-eqa-v0.json](open-eqa-v0.json) and the instructions below describe how to download the episode histories.

## Download Episode Histories

Episode histories are sourced form [HM3D](https://aihabitat.org/datasets/hm3d) and [ScanNet](http://www.scan-net.org).
Follow the instructions below to download the episode histories from both sources.

### HM3D

For HM3D episode histories, we provide two options: (1) directly download the RGB frames only. This is hosted by a third party repo (not by Meta), and may not be future proof, but can be faster or (2) extract the RGB, depth, and camera pose information from HM3D using the [Habitat](https://aihabitat.org/) simulator.

#### Option 1: Directly download RGB frames

The RGB frames for the HM3D episode histories are available in this [third party location](https://www.dropbox.com/scl/fi/t79gsjqlan8dneg7o63sw/open-eqa-hm3d-frames-v0.tgz?rlkey=1iuukwy2g3f5t06q4a3mxqobm) (12 Gb). You can use the following commands to download and extract the data:

```bash
wget -O open-eqa-hm3d-frames-v0.tgz <link above>
md5sum open-eqa-hm3d-frames-v0.tgz  # 286aa5d2fda99f4ed1567ae212998370
mkdir -p data/frames
tar -xzf open-eqa-hm3d-frames-v0.tgz -C data/frames
rm open-eqa-hm3d-frames-v0.tgz
```

Afterwards, your top-level directory structure should look like this:

```text
|- data
   |- frames
      |- hm3d-v0
         |- 000-hm3d-BFRyYbPCCPE
         |- ...
| - openeqa
| - ...
```

#### Option 2: Extract RGB, depth, and camera pose using Habitat

**Step 1:** Install the [Habitat](https://aihabitat.org/) simulator (version 2.5) by following the instructions [here](https://github.com/facebookresearch/habitat-sim#installation).

For example, on a headless server use:

```bash
conda install habitat-sim==0.2.5 headless -c conda-forge -c aihabitat
```

**Step 2:** Download the [HM3D](https://aihabitat.org/datasets/hm3d) validation data by first following the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) to get access to the dataset.

Then, use your api token-id and token-secret with this command:

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_val_v0.2
```

**Step 3:** Download the agent state information for the episode histories form [here](https://www.dropbox.com/scl/fi/wg1uj1gvr4tkcz9aq3tzb/open-eqa-hm3d-states-v0.tgz?rlkey=i69chnpib8ui4cfabxa3iy9oj).

```bash
wget -O open-eqa-hm3d-states-v0.tgz "https://www.dropbox.com/scl/fi/wg1uj1gvr4tkcz9aq3tzb/open-eqa-hm3d-states-v0.tgz?rlkey=i69chnpib8ui4cfabxa3iy9oj"
md5sum open-eqa-hm3d-states-v0.tgz  # 0db1a723573a5b9f5139e0df2ae960e0
mkdir -p data/frames
tar -xzf open-eqa-hm3d-states-v0.tgz -C data/frames
rm open-eqa-hm3d-states-v0.tgz
```

Afterwards, your directory structure should look like this:

```text
|- data
   |- frames
      |- hm3d-v0
   |- scene_datasets
      |- hm3d
| - openeqa
| - ...
```

**Step 4:** Extract frames from HM3D scenes.

You can either only extract RGB frames or extract RGB, depth, camera intrinsics, and camera pose information.

| Format | Size | Extraction Time |
| --- | --- | --- |
| RGB-only | 12 Gb | ~1 hrs|
| RGB-D + Intrinsics + Pose | 16Gb Gb | ~1.5 hrs|

To extract only the RGB frames, run:

```bash
python data/hm3d/extract-frames.py --rgb-only
```

To extract the RGB, depth, camera intrinsics, and camera pose information, run:

```bash
python data/hm3d/extract-frames.py
```

Afterwards, your top-level directory structure should look like this:

```text
|- data
   |- frames
      |- hm3d-v0
         |- 000-hm3d-BFRyYbPCCPE
         |- ...
   |- ...
| - openeqa
| - ...
```

### ScanNet

**Step 1:** Download [ScanNet](http://www.scan-net.org) by following the instructions [here](https://github.com/ScanNet/ScanNet#scannet-data).

Place the data in `data/raw/scannet`. Afterwards, your top-level directory should look like this:

```text
|- data
   |- raw
      |- scannet
         |- scans
            |- <scanId>
               |- <scanId>.sens
               |- ...
         |- scans_test
            |- <scanId>
               |- <scanId>.sens
               |- ...
|- openeqa
|- ...
```

**Step 2:** Extract episode histories $H$ from ScanNet scenes.

You can either only extract RGB frames or extract RGB, depth, camera intrinsics, and camera pose information.

| Format | Size | Extraction Time |
| --- | --- | --- |
| RGB-only | 62 Gb | ~8 hrs|
| RGB-D + Intrinsics + Pose | 70 Gb | ~10 hrs|

To extract only the RGB frames, run:

```bash
python data/scannet/extract-frames.py --rgb-only
```

To extract the RGB, depth, camera intrinsics, and camera pose information, run:

```bash
python data/scannet/extract-frames.py
```

Afterwards, your directory structure should look like this:

```text
|- data
   |- frames
      |- scannet-v0
         |- 002-scannet-scene0709_00
         |- ...
   |- raw
| - openeqa
| - ...
```

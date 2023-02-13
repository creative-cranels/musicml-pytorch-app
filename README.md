## Install dependencies

```bash
$ pip install 'datasets[audio]' yt-dlp musiclm-pytorch tensorboardX
```

## Run
```bash
$ python music.py 1 'the crystalline sounds of the piano in a ballroom'
```
**Parameters**

1. First parameter is flag to enable hubert download, use 0 
2. Second parameter is music title you want to generate

## Requirements

### Install ffmpeg on conda

#### Conda
```bash
$ conda install -c conda-forge ffmpeg
```

#### OSX
```bash
$ brew install ffmpeg
```
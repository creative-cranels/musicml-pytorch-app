import subprocess
import os
import requests
from pathlib import Path

from datasets import load_dataset, Audio

import torch
import torchaudio
from torch.utils.data import Dataset

from musiclm_pytorch import MuLaN, MuLaNTrainer, AudioSpectrogramTransformer, TextTransformer, MuLaNEmbedQuantizer, MusicLM
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
import sys

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='./tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'

def main(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.
    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            print(f"Working on {example['ytid']}..")
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )
            print(f"Result: {status}\nLOG: {log}\n\n")

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))

ds = main('./music_data', num_proc=2, limit=30, writer_batch_size=4) # change limit for larger dataset

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

class TextAudioDataset(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        ex = self.dset[idx]
        caption = ex['caption']
        wav, samplerate = torchaudio.load(ex['audio']['path'])
        return caption, wav

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# trainer = MuLaNTrainer(
#     mulan = mulan,
#     dataset = TextAudioDataset(ds),
#     batch_size = 3
# )

# trainer.train()

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# trainer = MuLaNTrainer(
#     mulan = mulan,
#     dataset = TextAudioDataset(ds),
#     batch_size = 3
# )

# trainer.train()

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)

def download_file(url, file_name):
    with open(file_name, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

def get_hubert():
    # Create a folder called hubert
    os.mkdir("hubert")

    # Download the files
    download_file("https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt", "hubert/hubert_base_ls960.pt")
    download_file("https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin", "hubert/hubert_base_ls960_L9_km500.bin")

download_hubert = sys.argv[1]
if download_hubert == "1":
  get_hubert()

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
)#.cuda()

# trainer = SemanticTransformerTrainer(
#     transformer = semantic_transformer,
#     wav2vec = wav2vec,
#     audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
#     folder ='./music_data',
#     batch_size = 1,
#     data_max_length = 320 * 32,
#     num_train_steps = 1
# )

# trainer.train()

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

# trainer = SoundStreamTrainer(
#     soundstream,
#     folder = './music_data',
#     batch_size = 4,
#     grad_accum_every = 8,         # effective batch size of 32
#     data_max_length = 320 * 32,
#     save_results_every = 2,
#     save_model_every = 4,
#     num_train_steps = 9
# )#.cuda()

# trainer.train()


wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

# soundstream = SoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
# )

# soundstream.load("./results/soundstream.8.pt")

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

# trainer = CoarseTransformerTrainer(
#     transformer = coarse_transformer,
#     soundstream = soundstream,
#     audio_conditioner = quantizer,
#     wav2vec = wav2vec,
#     folder = './music_data',
#     batch_size = 1,
#     data_max_length = 320 * 32,
#     save_results_every = 2,
#     save_model_every = 4,
#     num_train_steps = 9
# )
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

# trainer.train()

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 1024,
    depth = 6,
    audio_text_condition = True
)

# trainer = FineTransformerTrainer(
#     transformer = fine_transformer,
#     soundstream = soundstream,
#     audio_conditioner = quantizer,
#     folder = './music_data',
#     batch_size = 1,
#     data_max_length = 320 * 32,
#     num_train_steps = 9
# )
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

# trainer.train()

from musiclm_pytorch import MusicLM

audiolm = AudioLM(
    wav2vec = wav2vec,
    soundstream = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)
musiclm = MusicLM(
    audio_lm = audiolm,
    mulan_embed_quantizer = quantizer
)

music = musiclm(['the crystalline sounds of the piano in a ballroom']) # torch.Tensor

output_path = "out.wav"
sample_rate = 44100
torchaudio.save(output_path, music.cpu(), sample_rate)
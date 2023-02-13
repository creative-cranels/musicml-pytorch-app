import torch

from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer, MuLaNEmbedQuantizer, MusicLM
from audiolm_pytorch import AudioLM, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, SoundStream, CoarseTransformer, FineTransformer

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

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)
# get a ton of <sound, text> pairs and train

wavs = torch.randn(2, 1024)
conds = quantizer(wavs = wavs, namespace = 'semantic') # (2, 8, 1024) - 8 is number of quantizers
texts = torch.randint(0, 20000, (2, 256))

loss = mulan(wavs, texts)
loss.backward()

# after much training, you can embed sounds and text into a joint embedding space
# for conditioning the audio LM

embeds = mulan.get_audio_latents(wavs)  # during training
embeds = mulan.get_text_latents(texts)  # during inference

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
)

# trainer = SemanticTransformerTrainer(
#     transformer = semantic_transformer,
#     wav2vec = wav2vec,
#     audio_conditioner = quantizer,   # pass in the MulanEmbedQuantizer instance above
#     folder ='./audio-files',
#     batch_size = 1,
#     data_max_length = 320 * 32,
#     num_train_steps = 1
# )

# trainer.train()


###### AudioML
# soundstream = SoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
#     attn_window_size = 128,       # local attention receptive field at bottleneck
#     attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
# )

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
    attn_window_size = 128,       # local attention receptive field at bottleneck
    attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

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

music = musiclm(['the dubstep music inspired by japaneese anime']) # torch.Tensor
import argparse
import json
import os
import torch
from tqdm import tqdm
from plotly import graph_objects as go
from scipy.io import wavfile
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
import argparse
import multiprocessing
import os
import re
from pathlib import Path
from typing import List

import regex
import scipy.io.wavfile as wav

from num2words import num2words

from nemo.collections import asr as nemo_asr

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    NEMO_NORMALIZATION_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NEMO_NORMALIZATION_AVAILABLE = False
    
from IPython.display import Audio
import json

import numpy as np
from scripts_ctc.normalization_helpers import LATIN_TO_RU, RU_ABBREVIATIONS
from scripts_ctc.utils import get_segments, listener_configurer, listener_process, worker_configurer, worker_process

import os
from pathlib import PosixPath
from typing import List, Tuple, Union

import ctc_segmentation as cs
import numpy as np


class CTCSegment:
    
    def __init__(self, 
                 window_size = 8000,  
                 stride = 2.11,
                 sample_rate = 16000,
                 model_path = None, 
                 vocabulary = None):
        
        """
        Класс для выравнивания текста и аудио. 
        
        Parameters: 
        
        window_size - Размер окна CTC-alignment (Default: 8000)
        stride - параметр для сжатия окон CTC. Рекомендуемый диапазон: [2, 2.15]. Default: 2.11
        sample_rate - 
        model_path - путь к модели CTC
        vocabulary - словарь. Если не задан, используется словарь из модели
        
        Methods:
        
        predict_ctc - Получить log_probs с помощью используемой asr_model
        get_segments - Получить сегментацию
        process_file - Обработка одного файла 
        process_folder - Обработка всех файлов в manifest
        timit - Обработка файла в формате датасета timit (сегментация как для слов, так и для букв)
        timit_folder - Обработка всех файлов в manifest в формате датасета timit
        
        При написании в качестве источника использованы следующие ресурсы: 
        
        https://github.com/lumaku/ctc-segmentation
        https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tools/CTC_Segmentation_Tutorial.ipynb
        
        """
        
        if model_path is None and vocabulary is None:
            raise ValueError('Pass model or vocabulary')
        elif model_path is None and vocabulary is not None:
            self.vocabulary = vocabulary
        elif model_path is not None:
            self.get_model(model_path)
            
        self.window_size = window_size
        self.frame_duration_ms = 20
        self.stride = stride
        self.sample_rate = sample_rate
        
    def get_model(self, model_path):
        
        """
        Получение модели
        
        """
        
        if os.path.exists(model_path):
            self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
        elif model_path in nemo_asr.models.EncDecCTCModel.get_available_model_names():
            self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_path, strict=False)
        else:
            raise ValueError(
                f'{model_path} not a valid model name or path. Provide path to the pre-trained checkpoint '
                f'or choose from {nemo_asr.models.EncDecCTCModel.list_available_models()}'
            )

        self.vocabulary = self.asr_model.cfg.decoder.vocabulary
        self.vocabulary = ["ε"] + list(self.vocabulary)
        
    def jsonl_to_json(self, manifest):
        
        """
        Преобразование jsonl в json (для файла manifest)
        
        """
        
        manifest_json = manifest.repalce('.jsonl', '.json')
        
        with open(manifest, 'r') as json_file:
            data = [json.loads(line) for line in json_file]

        with open(manifest_json, 'w') as json_file:
            json.dump(data, json_file)
        
    def predict_ctc(self, audio_file):
        
        """
        Получение прогноза с помощью модели asr_model
        
        """
        sampling_rate, signal = wav.read(audio_file)
        log_probs = self.asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1, logprobs=True)[0]
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
        return log_probs, sampling_rate
    
    def get_segments(self, log_probs,
                            processed_text,
                            path_wav,
                            output_file, 
                            output_file_frames, 
                            sampling_rate):
        
        """
        Получение сегментов
        
        log_probs - Log_proba, полученные с помощью модели asr_model
        processed_text - Преобработанный (разбитый на токены) текст
        path_wav - Путь в аудио файлу
        output_file - Путь к файлу с сегментацией (по секундам)
        output_file_frames - Путь к файлу с сегментацией (по фреймам)
        
        """
        
        # Задаем параметры ctc_segmentation
        config = cs.CtcSegmentationParameters()
        config.char_list = self.vocabulary
        config.min_window_size = self.window_size
        config.frame_duration_ms = self.frame_duration_ms
        config.blank = config.space
        config.subsampling_factor = 2
        
        ground_truth_mat, utt_begin_indices = cs.prepare_text(config, processed_text)
        timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
        segments = cs.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, processed_text)
        segments = [list(segments[i])+[processed_text[i]] for i in range(len(segments))]
        
        segments_frames = []
        for segm in segments:
            segments_frames.append([round((segm[0]/self.stride)*sampling_rate), 
                                    round((segm[1]/self.stride)*sampling_rate), segm[-1]])
        
        with open(str(output_file), "w") as outfile:
#             outfile.write(str(path_wav) + "\n")

            for i, (start, end, score, tt) in enumerate(segments):
                outfile.write(
                    f'{start/self.stride} {end/self.stride} {score} | {tt} \n'
                )
                
        with open(str(output_file_frames), "w") as outfile:
#             outfile.write(str(path_wav) + "\n")

            for i, (start, end, tt) in enumerate(segments_frames):
                if {processed_text[i]} != ' ':
                    outfile.write(
                        f'{start} {end} | {tt} \n'
                    )
                
        return segments, segments_frames
    
    def timit(self, audio_file, text, output_dir):
        
        """
        Сегментация аудио файла и предобработка в формате timit
        
        audio_file - Путь к аудио-файлу
        text - Текст транскрипции
        output_dir - Директория для записи сегментаций
        
        """
        
        log_pr, sampling_rate = self.predict_ctc(audio_file)
        ids = audio_file
        
        segm_char, segm_frames_char = self.process_file(log_pr, text, output_dir,
                                                        logs = True, preprocess = 'chars', 
                                                        segment_d = '/segments_chars', 
                                                        display = False, ids = ids,
                                                        sampling_rate = sampling_rate)
        segm_words, segm_frames_words = self.process_file(log_pr, text, output_dir,
                                                          logs = True, preprocess = 'words', 
                                                          segment_d = '/segments_words', 
                                                          display = False, ids = ids,
                                                          sampling_rate = sampling_rate)
        
        return segm_frames_char, segm_frames_words
    
    def process_file(self, audio_file, text, output_dir, 
                     display = True, preprocess = 'words', 
                     logs = False, segment_d = '/segments',
                     ids = None, sampling_rate = None):
        
        """
        Обработка одного файла:
        
        Parameters:
        
        audio_file - Путь до аудио файла
        text - Транскрипция
        output_dir - Директория для записи сегментаций
        display - Выводить ли сегментацию на экран
        preprocess - Способ предобработки:
            Words - Сегментация по словам
            Chars - Сегментация по буквам
            Int - Сегментация по сочетаниям букв (2 - по биграммам)
        logs - Подаются ли в функцию прогнозы asr_model. Если да, то также требуется подать ids - название аудио файла (в дальнейшем будет использовано для формирования названия файлов с сегментацией)
        ids - Используется только при logs = True. Название аудио файла
        segment_d - Постфикс для папки с сегментацией. Default: /segments
        
        """
#         print(text)
        if len(re.sub('[^\w]', '', text)) == 0:
            segm = []
            segments_frames = []
        
        else:
            text = re.sub(' +', ' ', text).strip()
            
            if not logs:
                log_pr, sampling_rate = self.predict_ctc(audio_file)
                ids = audio_file.split('/')[-1].replace('.wav', '')
                path_wav = audio_file
            else:
                log_pr = audio_file
                path_wav = ids
                ids = ids.split('/')[-1].replace('.wav', '')

            segments_dir = output_dir+segment_d
            os.makedirs(segments_dir, exist_ok=True)

            if preprocess == 'words':
                processed_text = [k.strip() for k in text.split()]
            elif preprocess == 'chars':
                processed_text = list(text)
            else:
                processed_text = [''.join(list(text)[num:num+self.preprocess]) for num in range(0, len(list(text)), preprocess)]

            segment_file = segments_dir+'/'+ids+'.txt'
            segment_file_frames = segments_dir+'/'+ids+'_frames.txt'

            segm, segments_frames = self.get_segments(
                    log_probs = log_pr,
                    processed_text = processed_text,
                    path_wav = path_wav,
                    output_file = segment_file,
                    output_file_frames = segment_file_frames,
                    sampling_rate = sampling_rate
                )

            if display:
                cut_dir = output_dir+'/cut'
                os.makedirs(cut_dir, exist_ok=True)

                infos_high = self.process_alignment(segment_file, 
                                                    processed_text, 
                                                    output_dir = cut_dir, 
                                                    audio_file = path_wav)
                cut_file = cut_dir+'/'+ids+'_high_score_manifest.json'      
                self.display(cut_file)
            
        return segm, segments_frames
        
        
    def process_folder(self, manifest,
                       output_dir, preprocess = 'words', segment_d = '/segments'):
        
        """
        Обработка всех файлов в manifest
        
        """
        
        segments_dir = output_dir+'/segments'
        os.makedirs(segments_dir, exist_ok=True)
        
        with open(manifest, 'r') as file:
            lines = json.load(file)

        audio_files = [output_dir+line['audio_filepath'] for line in lines]
        texts = [line['text'] for line in lines]
    
        print(f"Has {len(audio_files)} files")
        for i, audio_file in enumerate(audio_files):
            print(audio_file)
            segm, segments_frames = self.process_file(audio_file, texts[i], output_dir, display = False, 
                                                      preprocess = preprocess, 
                                                      logs = False, segment_d = segment_d,
                                                      ids = None, sampling_rate = None)
            
    def timit_folder(self, manifest,
                       output_dir):
        
        """
        Обработка всех файлов в manifest
        
        """
        
        segments_dir = output_dir+'/segments'
        os.makedirs(segments_dir, exist_ok=True)
        
        with open(manifest, 'r') as file:
            lines = json.load(file)

        audio_files = [output_dir+line['audio_filepath'] for line in lines]
        texts = [line['text'] for line in lines]
    
        print(f"Has {len(audio_files)} files")
        for i, audio_file in enumerate(audio_files):
            print(audio_file)
            segm, segments_frames = self.timit(audio_file, texts[i], output_dir)
            
    def process_alignment(self, alignment_file, processed_text, output_dir, audio_file):
        
        """
        Нарезка аудио файла по сегментам
        
        alignment_file - Путь к файлу с сегментацией
        processed_text - Предобработанный текст
        output_dir - Директория для записи сегментаций
        
        
        """
        
        fragments_dir = os.path.join(output_dir, "high_score_clips")

        os.makedirs(fragments_dir, exist_ok=True)

        segments = []
        ref_text_processed = []
        with open(alignment_file, 'r') as f:
            for line in f:
                line = line.split('|')
                if len(line) == 1:
                    audio_file = line[0].strip()
                    continue
                ref_text_processed.append(line[1].strip())
                line = line[0].split()
                segments.append((float(line[0]), float(line[1]), float(line[2])))

        sampling_rate, signal = wavfile.read(audio_file)

        base_name = os.path.basename(alignment_file).replace('.txt', '')
        high_score_manifest = f'{base_name}_high_score_manifest.json'

        low_score_dur = 0
        high_score_dur = 0
        infos_high = []

        for i, (st, end, score) in enumerate(segments):
            segment = signal[round(st * sampling_rate) : round(end * sampling_rate)]
            duration = len(segment) / sampling_rate
            if duration > 0:
                text_processed = ref_text_processed[i].strip()
                high_score_dur += duration
                audio_filepath = os.path.join(fragments_dir, f'{base_name}_{i:04}.wav')
                info = {
                'audio_filepath': audio_filepath,
                'duration': duration,
                'text': text_processed,
                'score': round(score, 2)
                }
                infos_high.append(info)

                wavfile.write(audio_filepath, sampling_rate, segment)

        with open(os.path.join(output_dir, high_score_manifest), 'w', encoding='utf8') as f:
            json.dump(infos_high, f)

        return infos_high
        
    def plot_signal(self, signal):
        
        """ Plot the signal in time domain """
        fig_signal = go.Figure(
            go.Scatter(x=np.arange(signal.shape[0])/self.sample_rate,
                       y=signal, line={'color': 'green'},
                       name='Waveform',
                       hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>'),
            layout={
                'height': 200,
                'xaxis': {'title': 'Time, s'},
                'yaxis': {'title': 'Amplitude'},
                'title': 'Audio Signal',
                'margin': dict(l=0, r=0, t=40, b=0, pad=0),
            }
        )
        fig_signal.show()
        
        
    def plot_signal(self, signal, sample_rate):
        """ Plot the signal in time domain """
        fig_signal = go.Figure(
            go.Scatter(x=np.arange(signal.shape[0])/sample_rate,
                       y=signal, line={'color': 'green'},
                       name='Waveform',
                       hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>'),
            layout={
                'height': 200,
                'xaxis': {'title': 'Time, s'},
                'yaxis': {'title': 'Amplitude'},
                'title': 'Audio Signal',
                'margin': dict(l=0, r=0, t=40, b=0, pad=0),
            }
        )
        fig_signal.show()
    
    def display(self, manifest):
        """ Display audio and reference text."""

        with open(manifest, 'r') as file:
            lines = json.load(file)

        for sample in lines:
            print(sample)
            sample_rate, signal = wav.read(sample['audio_filepath'])
            self.plot_signal(signal, sample_rate)
            display(Audio(sample['audio_filepath']))
            display('Reference text:       ' + sample['text'])
            print('\n' + '-' * 110)
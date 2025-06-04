# ASR Pipeline para LibriSpeech con Wav2Vec2
# Autor: Assistant
# Descripción: Pipeline completo para entrenamiento de reconocimiento de fonemas

import os
import json
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset, DataLoader
import librosa
from pathlib import Path
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# ====== 1. CONFIGURACIÓN INICIAL ======
class Config:
    # Rutas de datos
    LIBRISPEECH_PATH = "./LibriSpeech/dev-clean"
    OUTPUT_DIR = "./output"
    MODEL_DIR = "./models"
    
    # Parámetros de audio
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 16.0  # segundos
    
    # Parámetros de entrenamiento
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500
    
    # Modelo base
    MODEL_NAME = "facebook/wav2vec2-base"

# ====== 2. PREPARACIÓN DE DATOS ======
class LibriSpeechDataProcessor:
    def __init__(self, config):
        self.config = config
        self.phoneme_vocab = {}
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        
    def parse_alignment_files(self) -> List[Dict]:
        """Parsea los archivos de alineación de LibriSpeech"""
        alignment_data = []
        
        for root, dirs, files in os.walk(self.config.LIBRISPEECH_PATH):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    audio_path = json_path.replace('.json', '.flac')
                    
                    if os.path.exists(audio_path):
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            
                        # Extraer secuencia de fonemas
                        phonemes = self.extract_phonemes(data)
                        if phonemes:
                            alignment_data.append({
                                'audio_path': audio_path,
                                'phonemes': phonemes,
                                'duration': self.get_audio_duration(audio_path)
                            })
        
        return alignment_data
    
    def extract_phonemes(self, alignment_data) -> List[str]:
        """Extrae secuencia de fonemas de los datos de alineación"""
        phonemes = []
        
        if 'words' in alignment_data:
            for word in alignment_data['words']:
                if 'phones' in word:
                    for phone in word['phones']:
                        phoneme = phone['phoneme']
                        # Limpiar fonemas (remover números de estrés)
                        clean_phoneme = re.sub(r'\d+', '', phoneme)
                        
                        # Manejar tokens especiales
                        if clean_phoneme in ['spn', 'sil']:
                            clean_phoneme = 'SIL'
                        
                        phonemes.append(clean_phoneme)
        
        return phonemes
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Obtiene la duración del audio"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform.shape[1] / sample_rate
        except:
            return 0.0
    
    def build_phoneme_vocabulary(self, alignment_data: List[Dict]) -> Dict[str, int]:
        """Construye el vocabulario de fonemas"""
        all_phonemes = set()
        
        for item in alignment_data:
            all_phonemes.update(item['phonemes'])
        
        # Agregar tokens especiales
        special_tokens = ['<pad>', '<unk>', '<blank>']
        phoneme_list = special_tokens + sorted(list(all_phonemes))
        
        self.phoneme_to_id = {phone: idx for idx, phone in enumerate(phoneme_list)}
        self.id_to_phoneme = {idx: phone for phone, idx in self.phoneme_to_id.items()}
        
        return self.phoneme_to_id
    
    def create_manifest(self, alignment_data: List[Dict]) -> pd.DataFrame:
        """Crea el archivo manifest con los datos procesados"""
        manifest_data = []
        
        for item in alignment_data:
            if item['duration'] > 0 and item['duration'] <= self.config.MAX_AUDIO_LENGTH:
                phoneme_ids = [self.phoneme_to_id.get(p, self.phoneme_to_id['<unk>']) 
                              for p in item['phonemes']]
                
                manifest_data.append({
                    'audio_path': item['audio_path'],
                    'phonemes': ' '.join(item['phonemes']),
                    'phoneme_ids': phoneme_ids,
                    'duration': item['duration']
                })
        
        return pd.DataFrame(manifest_data)

# ====== 3. DATASET PERSONALIZADO ======
class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_df, processor, config):
        self.manifest = manifest_df
        self.processor = processor
        self.config = config
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Cargar audio
        audio_path = row['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resamplear si es necesario
        if sample_rate != self.config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convertir a mono si es necesario
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Procesar audio
        audio_array = waveform.squeeze().numpy()
        inputs = self.processor(audio_array, sampling_rate=self.config.SAMPLE_RATE, 
                               return_tensors="pt", padding=True)
        
        # Preparar labels (phoneme IDs)
        phoneme_ids = eval(row['phoneme_ids']) if isinstance(row['phoneme_ids'], str) else row['phoneme_ids']
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': torch.tensor(phoneme_ids, dtype=torch.long)
        }

# ====== 4. COLLATOR PARA BATCH PROCESSING ======
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features):
        # Separar input values y labels
        input_features = [f['input_values'] for f in features]
        label_features = [f['labels'] for f in features]
        
        # Padding para audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Padding para labels
        max_label_length = max(len(labels) for labels in label_features)
        padded_labels = []
        
        for labels in label_features:
            padded = torch.full((max_label_length,), -100, dtype=torch.long)
            padded[:len(labels)] = labels
            padded_labels.append(padded)
        
        batch['labels'] = torch.stack(padded_labels)
        
        return batch

# ====== 5. ENTRENADOR PERSONALIZADO ======
class ASRTrainer:
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model_and_processor(self, vocab_size):
        """Configura el modelo y procesador"""
        # Crear tokenizer personalizado
        vocab_dict = {f"<phone_{i}>": i for i in range(vocab_size)}
        
        # Crear procesador
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.config.SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_dict,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|"
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        
        # Cargar modelo base
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.config.MODEL_NAME,
            vocab_size=vocab_size,
            ctc_loss_reduction="mean",
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Congelar feature extractor
        self.model.freeze_feature_extractor()
    
    def train(self, train_dataset, eval_dataset=None):
        """Entrena el modelo"""
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            group_by_length=True,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps" if eval_dataset else "no",
            num_train_epochs=self.config.NUM_EPOCHS,
            fp16=torch.cuda.is_available(),
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            save_total_limit=2,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
        )
        
        trainer.train()
        
        # Guardar modelo
        trainer.save_model(self.config.MODEL_DIR)
        self.processor.save_pretrained(self.config.MODEL_DIR)

# ====== 6. EVALUACIÓN ======
class ASREvaluator:
    def __init__(self, model, processor, phoneme_vocab):
        self.model = model
        self.processor = processor
        self.phoneme_vocab = phoneme_vocab
        self.id_to_phoneme = {v: k for k, v in phoneme_vocab.items()}
    
    def compute_per(self, predictions, references):
        """Calcula Phoneme Error Rate (PER)"""
        total_phonemes = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            # Convertir IDs a fonemas
            pred_phonemes = [self.id_to_phoneme.get(p, '<unk>') for p in pred]
            ref_phonemes = [self.id_to_phoneme.get(r, '<unk>') for r in ref]
            
            # Calcular distancia de edición (Levenshtein)
            errors = self.levenshtein_distance(pred_phonemes, ref_phonemes)
            total_errors += errors
            total_phonemes += len(ref_phonemes)
        
        return total_errors / total_phonemes if total_phonemes > 0 else 0.0
    
    def levenshtein_distance(self, s1, s2):
        """Calcula distancia de Levenshtein entre dos secuencias"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

# ====== 7. PIPELINE PRINCIPAL ======
def main():
    """Función principal que ejecuta todo el pipeline"""
    print("🚀 Iniciando pipeline de ASR para LibriSpeech")
    
    # Configuración
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 1. Procesamiento de datos
    print("📊 Procesando datos de LibriSpeech...")
    processor = LibriSpeechDataProcessor(config)
    
    # Parsear archivos de alineación
    alignment_data = processor.parse_alignment_files()
    print(f"✅ Encontrados {len(alignment_data)} archivos de audio con alineación")
    
    # Construir vocabulario
    phoneme_vocab = processor.build_phoneme_vocabulary(alignment_data)
    print(f"✅ Vocabulario construido con {len(phoneme_vocab)} fonemas")
    
    # Crear manifest
    manifest_df = processor.create_manifest(alignment_data)
    print(f"✅ Manifest creado con {len(manifest_df)} muestras")
    
    # Guardar manifest
    manifest_df.to_csv(os.path.join(config.OUTPUT_DIR, 'manifest.csv'), index=False)
    
    # Guardar vocabulario
    with open(os.path.join(config.OUTPUT_DIR, 'phoneme_vocab.json'), 'w') as f:
        json.dump(phoneme_vocab, f, indent=2)
    
    # 2. Preparar datasets
    print("🔄 Preparando datasets...")
    
    # Dividir datos (80% train, 20% eval)
    train_size = int(0.8 * len(manifest_df))
    train_df = manifest_df[:train_size]
    eval_df = manifest_df[train_size:]
    
    # 3. Configurar modelo
    print("🤖 Configurando modelo...")
    trainer = ASRTrainer(config)
    trainer.setup_model_and_processor(len(phoneme_vocab))
    
    # Crear datasets
    train_dataset = LibriSpeechDataset(train_df, trainer.processor, config)
    eval_dataset = LibriSpeechDataset(eval_df, trainer.processor, config)
    
    print(f"✅ Dataset de entrenamiento: {len(train_dataset)} muestras")
    print(f"✅ Dataset de evaluación: {len(eval_dataset)} muestras")
    
    # 4. Entrenar modelo
    print("🏋️ Iniciando entrenamiento...")
    trainer.train(train_dataset, eval_dataset)
    
    # 5. Evaluación
    print("📈 Evaluando modelo...")
    evaluator = ASREvaluator(trainer.model, trainer.processor, phoneme_vocab)
    
    # Aquí puedes agregar código para evaluar el modelo en el conjunto de test
    
    print("🎉 Pipeline completado exitosamente!")
    print(f"📁 Modelo guardado en: {config.MODEL_DIR}")
    print(f"📊 Resultados en: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()

# ====== 8. SCRIPT DE INFERENCIA ======
def inference_example():
    """Ejemplo de cómo usar el modelo entrenado para inferencia"""
    config = Config()
    
    # Cargar modelo entrenado
    model = Wav2Vec2ForCTC.from_pretrained(config.MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(config.MODEL_DIR)
    
    # Cargar vocabulario
    with open(os.path.join(config.OUTPUT_DIR, 'phoneme_vocab.json'), 'r') as f:
        phoneme_vocab = json.load(f)
    
    id_to_phoneme = {v: k for k, v in phoneme_vocab.items()}
    
    def transcribe_audio(audio_path):
        # Cargar audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Procesar
        inputs = processor(waveform.squeeze().numpy(), 
                          sampling_rate=sample_rate, 
                          return_tensors="pt", 
                          padding=True)
        
        # Inferencia
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Decodificar
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = [id_to_phoneme.get(pid.item(), '<unk>') 
                   for pid in predicted_ids.squeeze()]
        
        return phonemes
    
    # Ejemplo de uso
    # audio_path = "path/to/your/audio.wav"
    # phonemes = transcribe_audio(audio_path)
    # print("Fonemas predichos:", phonemes)

# ====== 9. UTILIDADES ADICIONALES ======
def setup_environment():
    """Instala dependencias necesarias"""
    import subprocess
    import sys
    
    packages = [
        "torch",
        "torchaudio", 
        "transformers",
        "librosa",
        "pandas",
        "numpy",
        "datasets"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_librispeech():
    """Descarga el dataset LibriSpeech dev-clean"""
    import urllib.request
    import tarfile
    
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    filename = "dev-clean.tar.gz"
    
    print("Descargando LibriSpeech dev-clean...")
    urllib.request.urlretrieve(url, filename)
    
    print("Extrayendo archivo...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    print("✅ Descarga completada!")

# Para ejecutar el pipeline completo, descomenta la siguiente línea:
main()
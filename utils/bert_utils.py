import tensorflow as tf
import numpy as np
from transformers import TFBertForMaskedLM, BertTokenizer

class BERTimbau:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForMaskedLM.from_pretrained(model_name)
    
    def _mask_tokens(self, inputs, mlm_probability=0.15):
        labels = inputs["input_ids"].copy()

        # Create mask array
        probability_matrix = np.random.rand(*labels.shape)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        mask_array = (probability_matrix < mlm_probability) & ~special_tokens_mask

        # Replace 80% of masked tokens with [MASK]
        inputs["input_ids"][mask_array] = self.tokenizer.mask_token_id

        # For loss computation, only keep masked positions — others go to -100
        labels[~mask_array] = -100

        return inputs["input_ids"], inputs["attention_mask"], labels


    def fine_tune(self, inputs):
        encoded = self.tokenizer(
            inputs, return_tensors="np", padding="max_length",
            truncation=True, max_length=128
        )
        input_ids, attention_mask, labels = self._mask_tokens(
            encoded, mlm_probability=0.15
        )

        def gen():
            for i in range(len(input_ids)):
                yield ({
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                    }, labels[i],
                )
        
        train_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {
                    "input_ids": tf.TensorSpec(shape=(128,), dtype=tf.int32),
                    "attention_mask": tf.TensorSpec(shape=(128,), dtype=tf.int32),
                },
                tf.TensorSpec(shape=(128,), dtype=tf.int32),
            )
        ).batch(8)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer)
        self.model.fit(train_dataset, epochs=3)

    def save_to_folder(self, folder_path):
        self.model.save_pretrained(folder_path)
        self.tokenizer.save_pretrained(folder_path)

    def load_from_folder(self, folder_path):
        self.tokenizer = BertTokenizer.from_pretrained(folder_path)
        self.model = TFBertForMaskedLM.from_pretrained(folder_path, from_pt=False)

    def _sample_with_temperature(self, logits, temperature=1.0):
        if temperature == 0:
            return tf.argmax(logits).numpy()
    
        logits = logits / temperature
        probs = tf.nn.softmax(logits).numpy()
        return np.random.choice(len(probs), p=probs)

    def generate_bert_text(self, seed_text, max_new_tokens=10, temperature=1.0):
        text = seed_text.strip()
        
        for _ in range(max_new_tokens):
            # Add a [MASK] at the end
            input_text = text + " " + self.tokenizer.mask_token
            inputs = self.tokenizer(input_text, return_tensors="tf", max_length=128, truncation=True)
            mask_token_index = tf.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[0][1]
            
            logits = self.model(inputs).logits[0, mask_token_index]
            
            # Sample from softmax with temperature
            predicted_token_id = self._sample_with_temperature(
                logits, temperature=temperature
            )
            predicted_token = self.tokenizer.decode([predicted_token_id]).strip()
            
            text += " " + predicted_token
            
            # Optionally break on end punctuation
            if predicted_token in [".", "!", "?", self.tokenizer.sep_token]:
                break
        
        return text

if __name__ == "__main__":
    model_path = "./bertimbau-mlm-tf"
    model_name = "neuralmind/bert-base-portuguese-cased"
    
    texts = [
        "A inteligência artificial está mudando o mundo.",
        "O cachorro correu atrás da bola.",
        "Hoje o céu está azul e limpo.",
        "A música brasileira é rica e diversificada."
    ]
    model = BERTimbau(model_name)
    model.fine_tune(texts)
    output = model.generate_bert_text("Software", 15, 1.5)
    print(output)

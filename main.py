import os
import numpy as np
from PIL import Image, ImageOps
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color, Rectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.image import Image as KivyImage
from kivy.metrics import dp
from kivy.uix.modalview import ModalView

# --- TFLite Interpreter (bukan tensorflow.keras) ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback kalau dijalankan di PC saat testing desktop

def get_resource_path(filename):
    """Path yang aman dipakai baik di desktop maupun setelah di-package jadi APK."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# --- SETTING TAMPILAN WINDOWS (Untuk simulasi layar HP) ---
Window.size = (380, 720)

# --- LOAD MODEL TFLITE ---
MODEL_PATH = get_resource_path('model_aksara.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ambil tipe data & shape input yang diharapkan model
INPUT_DTYPE = input_details[0]['dtype']
INPUT_SHAPE = input_details[0]['shape']

class_names = [
    'a', 'a_nengen', 'ah', 'ai', 'an', 'ang', 'ar', 'au',
    'b', 'ba', 'bah', 'bai', 'ban', 'bang', 'bar', 'bau', 'be', 'bee', 'bi', 'bo', 'bu',
    'c', 'ca', 'cah', 'cai', 'can', 'cang', 'car', 'cau', 'ce', 'cee', 'ci', 'co', 'cu',
    'd', 'da', 'dah', 'dai', 'dan', 'dang', 'dar', 'dau', 'de', 'dee', 'di', 'do', 'du',
    'e', 'ee',
    'g', 'ga', 'gah', 'gai', 'gan', 'gang', 'gar', 'gau', 'ge', 'gee', 'gi', 'go', 'gu',
    'gh', 'gha', 'ghah', 'ghai', 'ghan', 'ghang', 'ghar', 'ghau', 'ghe', 'ghee', 'ghi', 'gho', 'ghu',
    'h', 'ha', 'hah', 'hai', 'han', 'hang', 'har', 'hau', 'he', 'hee', 'hi', 'ho', 'hu',
    'i',
    'j', 'ja', 'jah', 'jai', 'jan', 'jang', 'jar', 'jau', 'je', 'jee', 'ji', 'jo', 'ju',
    'k', 'ka', 'kah', 'kai', 'kan', 'kang', 'kar', 'kau', 'ke', 'kee', 'ki', 'ko', 'ku',
    'l', 'la', 'lah', 'lai', 'lan', 'lang', 'lar', 'lau', 'le', 'lee', 'li', 'lo', 'lu',
    'm', 'ma', 'mah', 'mai', 'man', 'mang', 'mar', 'mau', 'me', 'mee', 'mi', 'mo', 'mu',
    'n', 'na', 'nah', 'nai', 'nan', 'nang', 'nar', 'nau', 'ne', 'nee', 'ni', 'no', 'nu',
    'ng', 'nga', 'ngah', 'ngai', 'ngan', 'ngang', 'ngar', 'ngau', 'nge', 'ngee', 'ngi', 'ngo', 'ngu',
    'ny', 'nya', 'nyah', 'nyai', 'nyan', 'nyang', 'nyar', 'nyau', 'nye', 'nyee', 'nyi', 'nyo', 'nyu',
    'o',
    'p', 'pa', 'pah', 'pai', 'pan', 'pang', 'par', 'pau', 'pe', 'pee', 'pi', 'po', 'pu',
    'r', 'ra', 'rah', 'rai', 'ran', 'rang', 'rar', 'rau', 're', 'ree', 'ri', 'ro', 'ru',
    's', 'sa', 'sah', 'sai', 'san', 'sang', 'sar', 'sau', 'se', 'see', 'si', 'so', 'su',
    't', 'ta', 'tah', 'tai', 'tan', 'tang', 'tar', 'tau', 'te', 'tee', 'ti', 'to', 'tu',
    'u',
    'w', 'wa', 'wah', 'wai', 'wan', 'wang', 'war', 'wau', 'we', 'wee', 'wi', 'wo', 'wu',
    'y', 'ya', 'yah', 'yai', 'yan', 'yang', 'yar', 'yau', 'ye', 'yee', 'yi', 'yo', 'yu'
]


# ==========================================
# CLASS KANVAS MENGGAMBAR
# ==========================================
class WritingCanvas(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 1, 1, 1)  # Latar putih permanen
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            with self.canvas:
                Color(0, 0, 0, 1)  # Tinta Hitam
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=6)

    def on_touch_move(self, touch):
        if self.collide_point(touch.x, touch.y) and 'line' in touch.ud:
            touch.ud['line'].points += [touch.x, touch.y]


# ==========================================
# MAIN APP KIVYMD
# ==========================================
class SkripsiApp(MDApp):
    def build(self):
        # --- TEMA MATERIAL DESIGN ---
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Brown"

        # Warna Custom Hex
        self.color_bg = (0.91, 0.75, 0.60, 1)        # Krem cokelat latar belakang
        self.color_card_top = (0.76, 0.49, 0.35, 1)  # Cokelat bata untuk header
        self.color_card_dark = (0.42, 0.23, 0.16, 1)  # Cokelat tua pekat

        screen = MDScreen()

        # Latar Belakang Utama
        with screen.canvas.before:
            Color(*self.color_bg)
            self.bg_rect = Rectangle(size=Window.size, pos=screen.pos)
        screen.bind(size=self._update_bg)

        # Layout Utama Deteksi (Tanpa Bottom Navigation)
        layout_deteksi = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # 1. Header Card (Gambar Siger)
        header_card = MDCard(
            radius=[dp(15), dp(15), dp(15), dp(15)],
            size_hint=(1, 0.2),
            elevation=0,
            md_bg_color=(1, 1, 1, 0)
        )
        bg_image = KivyImage(
            source=get_resource_path('siger-bg.jpg'),
            allow_stretch=True,
            keep_ratio=False
        )
        header_card.add_widget(bg_image)
        layout_deteksi.add_widget(header_card)

        # 2. Teks Instruksi
        lbl_sub = MDLabel(
            text="Please draw a character on the canvas",
            font_style="Caption",
            theme_text_color="Secondary",
            halign="center",
            size_hint_y=None,
            height=dp(30)
        )
        layout_deteksi.add_widget(lbl_sub)

        # 3. Area Kanvas Gambar (Card Putih)
        canvas_card = MDCard(
            md_bg_color=(1, 1, 1, 1),
            radius=[dp(20), dp(20), dp(20), dp(20)],
            size_hint=(1, 0.45),
            elevation=3,
            padding=dp(5)
        )
        self.kanvas = WritingCanvas()
        canvas_card.add_widget(self.kanvas)
        layout_deteksi.add_widget(canvas_card)

        # 4. Label Hasil Prediksi (Card Cokelat Tua)
        self.hasil_card = MDCard(
            md_bg_color=self.color_card_dark,
            radius=[dp(15), dp(15), dp(15), dp(15)],
            size_hint=(1, 0.15),
            elevation=2,
            padding=dp(10)
        )
        self.lbl_hasil = MDLabel(
            text="Prediction Result: -",
            halign="center",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            font_style="Subtitle1",
            bold=True
        )
        self.hasil_card.add_widget(self.lbl_hasil)
        layout_deteksi.add_widget(self.hasil_card)

        # 5. Tombol Aksi (Kamus, Hapus & Deteksi)
        layout_tombol = MDBoxLayout(
            orientation='horizontal',
            spacing=dp(10),
            size_hint=(1, 0.15),
            pos_hint={'center_x': 0.5}
        )

        btn_hapus = MDRaisedButton(
            text="CLEAR",
            md_bg_color=(0.8, 0.3, 0.3, 1),
            font_name='Roboto-Bold'
        )
        btn_hapus.bind(on_release=self.hapus_kanvas)
        
        # Tombol Baru untuk Memanggil Pop-up Kamus
        btn_kamus = MDRaisedButton(
            text="AKSARA CHART",
            md_bg_color=(0.4, 0.6, 0.4, 1), # Warna Hijau Kalem untuk pembeda
            font_name='Roboto-Bold'
        )
        btn_kamus.bind(on_release=self.tampilkan_kamus)

        btn_deteksi = MDRaisedButton(
            text="DETECT",
            md_bg_color=self.color_card_top,
            font_name='Roboto-Bold'
        )
        btn_deteksi.bind(on_release=self.proses_gambar)

        # Susunan tombol di bawah layar
        layout_tombol.add_widget(Widget()) # Spacer Kiri
        layout_tombol.add_widget(btn_hapus)
        layout_tombol.add_widget(btn_kamus)
        layout_tombol.add_widget(btn_deteksi)
        layout_tombol.add_widget(Widget()) # Spacer Kanan

        layout_deteksi.add_widget(layout_tombol)
        screen.add_widget(layout_deteksi)

        return screen

    def _update_bg(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos

    # ==========================================
    # LOGIKA POP-UP KAMUS
    # ==========================================
    def tampilkan_kamus(self, instance):
        # Menggunakan ukuran modal 95% lebar dan 85% tinggi layar
        self.modal = ModalView(size_hint=(0.95, 0.85), auto_dismiss=True, background_color=(0,0,0,0.7))
        
        box = MDBoxLayout(
            orientation='vertical', 
            md_bg_color=(1, 1, 1, 1), 
            radius=[15, 15, 15, 15], 
            padding=dp(5)
        )
        
        # Header kecil dengan tombol tutup silang (X)
        header = MDBoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        header.add_widget(Widget()) # Dorong tombol ke kanan
        btn_close = MDIconButton(icon="close-circle", on_release=self.modal.dismiss)
        header.add_widget(btn_close)
        box.add_widget(header)
        
        # Gambar responsif tanpa ScrollView
        img_tabel = KivyImage(
            source=get_resource_path('Tabel_Aksara.jpg'),
            size_hint=(1, 1),
            allow_stretch=True,
            keep_ratio=True
        )
        box.add_widget(img_tabel)
        
        self.modal.add_widget(box)
        self.modal.open()

    # ==========================================
    # LOGIKA PREDIKSI & KONTROL
    # ==========================================
    def hapus_kanvas(self, instance):
        self.kanvas.canvas.clear()
        with self.kanvas.canvas.before:
            Color(1, 1, 1, 1)
            self.kanvas.rect = Rectangle(pos=self.kanvas.pos, size=self.kanvas.size)
        self.lbl_hasil.text = "Canvas cleared. Ready to use."
        self.hasil_card.md_bg_color = self.color_card_dark

    def proses_gambar(self, instance):
        nama_file = os.path.join(self.user_data_dir, "input_temp.png")
        self.kanvas.export_to_png(nama_file)
        self.lbl_hasil.text = "Processing AI..."
        self.hasil_card.md_bg_color = (0.3, 0.3, 0.3, 1)  # Warna loading
        Clock.schedule_once(lambda dt: self.jalankan_prediksi(nama_file), 0.5)

    def jalankan_prediksi(self, file_path):
        try:
            # 1. Baca gambar menggunakan Pillow (Konversi ke Grayscale)
            img = Image.open(file_path).convert('L')

            # 2. Cari bounding box tulisan (potong area kosong)
            img_inverted = ImageOps.invert(img)
            bbox = img_inverted.getbbox() 

            if bbox is not None:
                crop = img.crop(bbox)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                # --- MANIPULASI MENIRU DATASET 50x50 ---
                max_crop_dim = max(w, h)
                scale = 30.0 / max_crop_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
                canvas_50 = Image.new('L', (50, 50), color=255)

                start_x = (50 - new_w) // 2
                start_y = (50 - new_h) // 2
                canvas_50.paste(crop_resized, (start_x, start_y))

                # Resize ke 64x64 (Ukuran input CNN)
                img_final_64 = canvas_50.resize((64, 64), Image.Resampling.LANCZOS)
                img_array = np.array(img_final_64)
                img_final = img_array.reshape(1, 64, 64, 1).astype(INPUT_DTYPE)

                # Inferensi pakai TFLite Interpreter
                interpreter.set_tensor(input_details[0]['index'], img_final)
                interpreter.invoke()
                prediksi = interpreter.get_tensor(output_details[0]['index'])

                idx = np.argmax(prediksi)
                hasil = class_names[idx]
                probabilitas = np.max(prediksi) * 100

                # ==========================================
                # FITUR DETEKSI ERROR / CORETAN ASAL
                # ==========================================
                BATAS_MINIMAL = 75.0

                if w < 10 or h < 10:
                    self.lbl_hasil.text = "Input is too small!\nPlease draw clearly."
                    self.hasil_card.md_bg_color = (0.8, 0.5, 0.2, 1)  # Oranye Peringatan

                elif probabilitas < BATAS_MINIMAL:
                    self.lbl_hasil.text = f"Unrecognized Character\n(Confidence too low: {probabilitas:.2f}%)"
                    self.hasil_card.md_bg_color = (0.8, 0.3, 0.3, 1)  # Merah Eror

                else:
                    self.lbl_hasil.text = f"Prediction: {hasil}\nConfidence: {probabilitas:.2f}%"
                    self.hasil_card.md_bg_color = (0.2, 0.6, 0.3, 1)  # Hijau Sukses

            else:
                self.lbl_hasil.text = "Canvas is still empty!"
                self.hasil_card.md_bg_color = (0.8, 0.3, 0.3, 1)
        except Exception as e:
            self.lbl_hasil.text = f"Error: {e}"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == '__main__':
    SkripsiApp().run()
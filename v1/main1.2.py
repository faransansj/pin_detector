import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
import threading

class OptimizedImageProcessor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("최적화된 이미지 처리")
        self.window.geometry("1200x600")

        # 이미지 최대 크기 제한
        self.MAX_IMAGE_SIZE = 800
        
        # 스레드 풀 생성
        self.executor = ThreadPoolExecutor(max_workers=4)

        # UI 구성
        self.setup_ui()

    def setup_ui(self):
        # 이미지 표시 프레임
        self.image_frame = tk.Frame(self.window)
        self.image_frame.pack(pady=10)

        # 캔버스 생성
        self.canvases = {}
        titles = ["원본 이미지", "Otsu 이진화", "Zhang-Suen 스켈레톤화"]
        for i, title in enumerate(titles):
            canvas = tk.Canvas(self.image_frame, width=350, height=350)
            canvas.grid(row=0, column=i, padx=10)
            tk.Label(self.image_frame, text=title).grid(row=1, column=i)
            self.canvases[title] = canvas

        # 진행 상태 표시
        self.progress_label = tk.Label(self.window, text="준비")
        self.progress_label.pack(pady=5)

        # 버튼
        self.load_button = tk.Button(self.window, text="이미지 로드", command=self.start_processing)
        self.load_button.pack(pady=10)

    def update_progress(self, text):
        """스레드 안전한 진행 상태 업데이트"""
        self.window.after(0, lambda: self.progress_label.config(text=text))

    def resize_image(self, image):
        """이미지 크기 제한"""
        height, width = image.shape[:2]
        if width > self.MAX_IMAGE_SIZE or height > self.MAX_IMAGE_SIZE:
            if width > height:
                new_width = self.MAX_IMAGE_SIZE
                new_height = int(height * (self.MAX_IMAGE_SIZE/width))
            else:
                new_height = self.MAX_IMAGE_SIZE
                new_width = int(width * (self.MAX_IMAGE_SIZE/height))
            return cv2.resize(image, (new_width, new_height))
        return image

    def optimized_zhang_suen(self, img):
        """최적화된 Zhang-Suen 알고리즘"""
        # ROI 추출
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x, y, w, h = cv2.boundingRect(binary)
        roi = binary[y:y+h, x:x+w]

        # 결과 이미지 초기화
        skeleton = np.zeros_like(roi)
        
        # 이미지를 0과 1로 변환
        roi = roi // 255

        def neighbors_fast(img):
            """벡터화된 이웃 픽셀 계산"""
            rows, cols = img.shape
            P = np.zeros((rows, cols, 8), dtype=np.int32)
            
            # 8방향 이웃 계산을 벡터화
            P[:, :, 0] = np.roll(img,  1, axis=0)  # P1
            P[:, :, 1] = np.roll(np.roll(img,  1, axis=0),  1, axis=1)  # P2
            P[:, :, 2] = np.roll(img,  1, axis=1)  # P3
            P[:, :, 3] = np.roll(np.roll(img, -1, axis=0),  1, axis=1)  # P4
            P[:, :, 4] = np.roll(img, -1, axis=0)  # P5
            P[:, :, 5] = np.roll(np.roll(img, -1, axis=0), -1, axis=1)  # P6
            P[:, :, 6] = np.roll(img, -1, axis=1)  # P7
            P[:, :, 7] = np.roll(np.roll(img,  1, axis=0), -1, axis=1)  # P8
            
            return P

        def transitions_fast(P):
            """벡터화된 전이 횟수 계산"""
            return ((P[:,:,:-1] == 0) & (P[:,:,1:] == 1)).sum(axis=2)

        # 반복 처리
        while True:
            P = neighbors_fast(roi)
            B = P.sum(axis=2)
            T = transitions_fast(np.concatenate((P, P[:,:,:1]), axis=2))
            
            # 첫 번째 단계 조건
            D1 = ((B >= 2) & (B <= 6) & (T == 1) & 
                  (P[:,:,0] * P[:,:,2] * P[:,:,4] == 0) & 
                  (P[:,:,2] * P[:,:,4] * P[:,:,6] == 0))
            
            roi1 = roi.copy()
            roi1[D1] = 0
            
            # 두 번째 단계
            P = neighbors_fast(roi1)
            B = P.sum(axis=2)
            T = transitions_fast(np.concatenate((P, P[:,:,:1]), axis=2))
            
            D2 = ((B >= 2) & (B <= 6) & (T == 1) & 
                  (P[:,:,0] * P[:,:,2] * P[:,:,6] == 0) & 
                  (P[:,:,0] * P[:,:,4] * P[:,:,6] == 0))
            
            roi2 = roi1.copy()
            roi2[D2] = 0
            
            if np.array_equal(roi2, roi):
                break
            
            roi = roi2

        # 결과를 원래 크기로 복원
        skeleton[roi == 1] = 255
        result = np.zeros_like(binary)
        result[y:y+h, x:x+w] = skeleton

        return result

    def process_image_chunk(self, chunk):
        """이미지 청크 처리"""
        return self.optimized_zhang_suen(chunk)

    def start_processing(self):
        """이미지 처리 시작"""
        try:
            # 파일 선택
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not file_path:
                return

            # 이미지 로드 및 크기 제한
            self.update_progress("이미지 로딩 중...")
            original = cv2.imread(file_path)
            if original is None:
                raise Exception("이미지를 불러올 수 없습니다.")

            original = self.resize_image(original)
            
            # 원본 이미지 표시
            self.display_image(original, self.canvases["원본 이미지"])

            # Otsu 이진화
            self.update_progress("이진화 처리 중...")
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 
                             self.canvases["Otsu 이진화"])

            # 이미지를 청크로 분할하여 병렬 처리
            self.update_progress("스켈레톤화 처리 중...")
            chunks = np.array_split(binary, 4)  # 4개의 청크로 분할
            futures = [self.executor.submit(self.process_image_chunk, chunk) 
                      for chunk in chunks]
            
            # 결과 결합
            results = [future.result() for future in futures]
            skeleton = np.vstack(results)

            # 결과 표시
            self.display_image(cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR), 
                             self.canvases["Zhang-Suen 스켈레톤화"])
            
            self.update_progress("처리 완료")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_progress("오류 발생")

    def display_image(self, cv_img, canvas):
        """이미지 표시"""
        # 캔버스 크기에 맞게 조정
        height, width = cv_img.shape[:2]
        max_size = 350
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size/width))
        else:
            new_height = max_size
            new_width = int(width * (max_size/height))

        resized = cv2.resize(cv_img, (new_width, new_height))
        
        # BGR에서 RGB로 변환
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(rgb_img)
        
        # PhotoImage로 변환
        photo = ImageTk.PhotoImage(pil_img)
        
        # 캔버스 크기 설정
        canvas.config(width=new_width, height=new_height)
        canvas.delete("all")
        canvas.create_image(new_width//2, new_height//2, image=photo)
        canvas.image = photo

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = OptimizedImageProcessor()
    app.run()

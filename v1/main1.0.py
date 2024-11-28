import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class CircuitLineDetector:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("회로도 선 검출기")
        self.window.geometry("1000x600")

        # 이미지 표시를 위한 캔버스 생성
        self.canvas_original = tk.Canvas(self.window, width=400, height=400)
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)
        
        self.canvas_result = tk.Canvas(self.window, width=400, height=400)
        self.canvas_result.grid(row=0, column=1, padx=10, pady=10)

        # 버튼 프레임
        button_frame = tk.Frame(self.window)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # 버튼 생성
        self.load_button = tk.Button(button_frame, text="이미지 로드", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.process_button = tk.Button(button_frame, text="선 검출", command=self.detect_lines)
        self.process_button.pack(side=tk.LEFT, padx=5)

        # 이미지 저장 변수
        self.original_img = None
        self.photo_original = None
        self.photo_result = None

    def load_image(self):
        try:
            # 파일 선택 대화상자
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            
            if file_path:
                # OpenCV로 이미지 읽기
                self.original_img = cv2.imread(file_path)
                if self.original_img is None:
                    raise Exception("이미지를 불러올 수 없습니다.")

                # 이미지 리사이즈
                height, width = self.original_img.shape[:2]
                if width > 400:
                    ratio = 400/width
                    self.original_img = cv2.resize(self.original_img, 
                                                 (400, int(height*ratio)))

                # 원본 이미지 표시
                self.display_image(self.original_img, self.canvas_original, "원본 이미지")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def detect_lines(self):
        try:
            if self.original_img is None:
                messagebox.showwarning("경고", "먼저 이미지를 로드해주세요.")
                return

            # 그레이스케일 변환
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            
            # 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 캐니 엣지 검출
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # PPHT로 선 검출
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                  threshold=50, 
                                  minLineLength=30, 
                                  maxLineGap=10)

            # 결과 이미지 생성
            result_img = self.original_img.copy()

            if lines is not None:
                # 각 선마다 랜덤한 색상으로 그리기
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    color = (np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255))
                    cv2.line(result_img, (x1, y1), (x2, y2), color, 2)

            # 결과 이미지 표시
            self.display_image(result_img, self.canvas_result, "처리된 이미지")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, cv_img, canvas, title):
        # OpenCV BGR을 RGB로 변환
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(rgb_img)
        
        # PhotoImage로 변환
        photo = ImageTk.PhotoImage(pil_img)
        
        # 캔버스 크기 설정
        canvas.config(width=pil_img.width, height=pil_img.height)
        
        # 이전 이미지 삭제
        canvas.delete("all")
        
        # 새 이미지 표시
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.create_text(pil_img.width//2, 20, text=title, fill="black")
        
        # 참조 보존
        if canvas == self.canvas_original:
            self.photo_original = photo
        else:
            self.photo_result = photo

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CircuitLineDetector()
    app.run()

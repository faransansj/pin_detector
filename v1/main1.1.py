import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class CircuitImageProcessor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("회로도 이미지 처리")
        self.window.geometry("1200x800")

        # 이미지를 표시할 프레임
        self.image_frame = tk.Frame(self.window)
        self.image_frame.pack(pady=10)

        # 각 단계별 이미지를 표시할 캔버스
        self.canvas_original = tk.Canvas(self.image_frame, width=350, height=350)
        self.canvas_original.grid(row=0, column=0, padx=10)
        
        self.canvas_skeleton = tk.Canvas(self.image_frame, width=350, height=350)
        self.canvas_skeleton.grid(row=0, column=1, padx=10)
        
        self.canvas_ppht = tk.Canvas(self.image_frame, width=350, height=350)
        self.canvas_ppht.grid(row=0, column=2, padx=10)

        # 제목 레이블
        tk.Label(self.image_frame, text="원본 이미지").grid(row=1, column=0)
        tk.Label(self.image_frame, text="스켈레톤화").grid(row=1, column=1)
        tk.Label(self.image_frame, text="PPHT 결과").grid(row=1, column=2)

        # 버튼 프레임
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)

        # 버튼 생성
        self.load_button = tk.Button(button_frame, text="이미지 로드", command=self.process_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # 이미지 저장 변수
        self.original_img = None
        self.skeleton_img = None
        self.ppht_img = None

    def skeletonize(self, img):
        # 이진화
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 스켈레톤화
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
                
        return skeleton

    def apply_ppht(self, skeleton_img):
        # PPHT 적용
        lines = cv2.HoughLinesP(
            skeleton_img, 
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        # 결과 이미지 생성
        result = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                color = (np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255))
                cv2.line(result, (x1, y1), (x2, y2), color, 2)
                
        return result

    def process_image(self):
        try:
            # 파일 선택
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            
            if file_path:
                # 원본 이미지 로드
                self.original_img = cv2.imread(file_path)
                if self.original_img is None:
                    raise Exception("이미지를 불러올 수 없습니다.")

                # 그레이스케일 변환
                gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
                
                # 스켈레톤화
                self.skeleton_img = self.skeletonize(gray)
                
                # PPHT 적용
                self.ppht_img = self.apply_ppht(self.skeleton_img)
                
                # 이미지 크기 조정 및 표시
                self.display_image(self.original_img, self.canvas_original, "원본")
                self.display_image(cv2.cvtColor(self.skeleton_img, cv2.COLOR_GRAY2BGR), 
                                 self.canvas_skeleton, "스켈레톤")
                self.display_image(self.ppht_img, self.canvas_ppht, "PPHT")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_image(self, cv_img, canvas, title):
        # 이미지 크기 조정
        height, width = cv_img.shape[:2]
        max_size = 350
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size/width))
        else:
            new_height = max_size
            new_width = int(width * (max_size/height))
            
        resized = cv2.resize(cv_img, (new_width, new_height))
        
        # OpenCV BGR을 RGB로 변환
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(rgb_img)
        
        # PhotoImage로 변환
        photo = ImageTk.PhotoImage(pil_img)
        
        # 캔버스 크기 설정
        canvas.config(width=new_width, height=new_height)
        
        # 이전 이미지 삭제
        canvas.delete("all")
        
        # 새 이미지 표시
        canvas.create_image(new_width//2, new_height//2, image=photo)
        canvas.image = photo  # 참조 보존

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CircuitImageProcessor()
    app.run()

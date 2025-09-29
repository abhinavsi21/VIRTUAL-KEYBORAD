import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import webbrowser
import math

class VirtualKeyboard:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        # set to comfortable resolution (change if needed)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # NOTE: changed 'BACK' -> 'BACKSPACE' so key names are consistent everywhere
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE', 'BACKSPACE'],
            ['GOOGLE', 'INSTAGRAM', 'YOUTUBE']
        ]

        self.caps_lock = False
        self.typed_text = ""
        self.last_click_time = 0
        self.click_delay = 0.35  # slightly faster response okay
        self.search_mode = None
        self.search_text = ""
        self.gesture_start_time = 0
        self.gesture_hold_time = 0.3
        self.is_gesturing = False
        self.current_gesture_key = None
        self.key_positions = []
        self.key_size = (60, 50)
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01

        # Stats & history: ensure we record timestamps to compute WPM
        self.stats = {
            'keys_pressed': 0,
            'session_start': time.time(),
            'hotkeys_used': 0,
            'accuracy': 100.0,
            'wpm': 0.0,
            'last_key_time': time.time()
        }
        # typing_history will store timestamps of recent character presses
        self.typing_history = []

        # colors & UI settings (unchanged)
        self.colors = {
            'background': (25, 25, 35),
            'key_normal': (45, 50, 65),
            'key_hover': (65, 75, 95),
            'key_pressed': (85, 95, 120),
            'text': (240, 245, 255),
            'text_secondary': (180, 185, 200),
            'accent_blue': (100, 150, 255),
            'accent_green': (100, 200, 150),
            'accent_purple': (150, 100, 255),
            'accent_orange': (255, 150, 100),
            'google': (66, 133, 244),
            'youtube': (255, 0, 0),
            'instagram': (225, 48, 108),
            'warning': (255, 193, 7),
            'success': (40, 167, 69),
            'danger': (220, 53, 69),
            'shadow': (15, 15, 25)
        }
        self.ui_elements = {
            'header_height': 80,
            'footer_height': 60,
            'sidebar_width': 250,
            'corner_radius': 8,
            'shadow_offset': 3
        }

        self.search_urls = {
            'GOOGLE': 'https://www.google.com/search?q=',
            'INSTAGRAM': 'https://www.instagram.com/explore/tags/',
            'YOUTUBE': 'https://www.youtube.com/results?search_query='
        }

    # ---------- drawing helpers (same behavior) ----------
    def create_overlay_background(self, frame, x, y, width, height, alpha=0.7):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (25, 25, 35), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_rounded_rectangle(self, frame, pt1, pt2, color, radius=8, thickness=-1):
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness == -1:
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.rectangle(frame, pt1, pt2, color, thickness)

    def draw_shadow(self, frame, x, y, width, height, offset=3):
        shadow_color = self.colors['shadow']
        self.draw_rounded_rectangle(
            frame,
            (x + offset, y + offset),
            (x + width + offset, y + height + offset),
            shadow_color,
            radius=self.ui_elements['corner_radius']
        )

    # ---------- improved key-positioning: center each row ----------
    def calculate_key_positions(self, frame_width, frame_height):
        self.key_positions = []
        margin = 30
        start_y = frame_height - 260
        key_spacing = 12
        for row_idx, row in enumerate(self.keys):
            # determine widths for each key in the row (special-casing SPACE and BACKSPACE)
            widths = []
            for k in row:
                if k == 'SPACE':
                    widths.append(int(self.key_size[0] * 2.8))
                elif k in ('BACKSPACE',):
                    widths.append(int(self.key_size[0] * 1.6))
                else:
                    widths.append(self.key_size[0])
            total_row_width = sum(widths) + key_spacing * (len(row) - 1)
            current_x = int((frame_width - total_row_width) // 2)  # center the row
            row_positions = []
            for idx, key in enumerate(row):
                w = widths[idx]
                row_positions.append({
                    'key': key,
                    'x': current_x,
                    'y': start_y + row_idx * (self.key_size[1] + 18),
                    'width': w,
                    'height': self.key_size[1]
                })
                current_x += w + key_spacing
            self.key_positions.append(row_positions)

    def get_key_color(self, key, is_hover, is_pressed):
        base_colors = {
            'GOOGLE': self.colors['google'],
            'YOUTUBE': self.colors['youtube'],
            'INSTAGRAM': self.colors['instagram'],
            'BACKSPACE': self.colors['warning'],
            'SPACE': self.colors['accent_blue']
        }
        color = base_colors.get(key, self.colors['key_normal'])
        if key == self.search_mode:
            color = tuple(min(255, c + 30) for c in color)
        if is_pressed:
            color = tuple(min(255, c + 50) for c in color)
        elif is_hover:
            color = tuple(min(255, c + 25) for c in color)
        return color

    def draw_enhanced_keyboard(self, frame, hover_key=None, pressed_key=None):
        if not self.key_positions:
            self.calculate_key_positions(frame.shape[1], frame.shape[0])
        for row in self.key_positions:
            for key_info in row:
                key = key_info['key']
                x = int(key_info['x'])
                y = int(key_info['y'])
                width = int(key_info['width'])
                height = int(key_info['height'])
                is_hover = hover_key == key
                is_pressed = pressed_key == key
                color = self.get_key_color(key, is_hover, is_pressed)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
                border_color = tuple(min(255, c + 40) for c in color)
                cv2.rectangle(overlay, (x, y), (x + width, y + height), border_color, 2)
                alpha = 0.45 if not is_hover else 0.65
                cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
                if is_hover or is_pressed:
                    highlight_color = tuple(min(255, c + 60) for c in color)
                    cv2.rectangle(frame, (x + 1, y + 1), (x + width - 1, y + height - 1), highlight_color, 1)
                self.draw_key_text(frame, key, x, y, width, height, is_hover)

    def draw_key_text(self, frame, key, x, y, width, height, is_hover=False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(key) <= 1:
            font_scale = 0.6
            thickness = 2
        elif len(key) <= 5:
            font_scale = 0.4
            thickness = 1
        else:
            font_scale = 0.35
            thickness = 1
        text_color = self.colors['text']
        if key in ['GOOGLE', 'INSTAGRAM', 'YOUTUBE']:
            text_color = (255, 255, 255)
        elif is_hover:
            text_color = tuple(min(255, c + 20) for c in text_color)
        text_size = cv2.getTextSize(key, font, font_scale, thickness)[0]
        text_x = int(x + (width - text_size[0]) // 2)
        text_y = int(y + (height + text_size[1]) // 2)
        shadow_color = (0, 0, 0)
        cv2.putText(frame, key, (text_x + 1, text_y + 1), font, font_scale, shadow_color, thickness)
        cv2.putText(frame, key, (text_x, text_y), font, font_scale, text_color, thickness)

    def draw_modern_header(self, frame):
        h, w = frame.shape[:2]
        header_height = self.ui_elements['header_height']
        self.create_overlay_background(frame, 0, 0, w, header_height, 0.8)
        cv2.line(frame, (0, header_height), (w, header_height), self.colors['accent_blue'], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "AI Virtual Keyboard"
        cv2.putText(frame, title, (20, 35), font, 0.8, self.colors['text'], 2)
        status_x = w - 360
        if self.search_mode:
            status_text = f"Search: {self.search_mode}"
            status_color = self.get_key_color(self.search_mode, False, False)
        else:
            status_text = "Ready"
            status_color = self.colors['success']
        cv2.putText(frame, status_text, (status_x, 35), font, 0.5, status_color, 1)

    # draw circular progress when holding gesture
    def draw_gesture_feedback(self, frame, finger_pos, hover_key):
        if not finger_pos or not self.is_gesturing:
            return
        x, y = finger_pos
        if self.current_gesture_key == hover_key:
            progress = (time.time() - self.gesture_start_time) / self.gesture_hold_time
            progress = min(1.0, progress)
            cv2.circle(frame, (x, y), 20, (50, 50, 50), 3)
            angle = int(360 * progress)
            if angle > 0:
                for i in range(0, angle, 5):
                    rad = math.radians(i - 90)
                    x1 = int(x + 17 * math.cos(rad))
                    y1 = int(y + 17 * math.sin(rad))
                    cv2.circle(frame, (x1, y1), 2, self.colors['accent_green'], -1)
            center_color = self.colors['success'] if progress >= 1.0 else self.colors['warning']
            cv2.circle(frame, (x, y), 5, center_color, -1)

    def draw_hand_skeleton(self, frame, landmarks, frame_shape):
        if not landmarks:
            return
        h, w = frame_shape[:2]
        overlay = frame.copy()
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(overlay, start_point, end_point, (0, 255, 255), 2)
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if idx in [4, 8, 12, 16, 20]:
                color = (0, 255, 0)
                radius = 4
            elif idx == 0:
                color = (255, 0, 0)
                radius = 6
            else:
                color = (255, 255, 0)
                radius = 3
            cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

    def draw_typed_text_display(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.search_mode:
            display_text = f"{self.search_mode}: {self.search_text}"
            text_color = (0, 255, 255)
        else:
            display_text = f"Typed: {self.typed_text}"
            text_color = (255, 255, 255)
        if len(display_text) > 60:
            display_text = display_text[:57] + "..."
        text_size = cv2.getTextSize(display_text, font, 0.7, 2)[0]
        bg_width = text_size[0] + 20
        bg_height = text_size[1] + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + bg_width, 10 + bg_height), (100, 100, 100), 2)
        cv2.putText(frame, display_text, (20, 35), font, 0.7, text_color, 2)

    # ---------- WPM calculation ----------
    def calculate_wpm(self):
        now = time.time()
        # keep only the last 30 seconds of timestamps for WPM smoothing
        window = 30.0
        self.typing_history = [t for t in self.typing_history if now - t <= window]
        chars_in_window = len(self.typing_history)
        if chars_in_window <= 1:
            self.stats['wpm'] = 0.0
            return
        # chars per second -> wpm (5 chars per word standard)
        time_span = max(0.1, now - self.typing_history[0])
        chars_per_sec = chars_in_window / time_span
        self.stats['wpm'] = (chars_per_sec * 60) / 5

    def get_finger_tip(self, landmarks, frame_shape):
        if landmarks:
            index_tip = landmarks[8]
            h, w = frame_shape[:2]
            return int(index_tip.x * w), int(index_tip.y * h)
        return None

    def get_hovered_key(self, finger_pos):
        if not finger_pos or not self.key_positions:
            return None
        x, y = finger_pos
        for row in self.key_positions:
            for key_info in row:
                kx, ky = int(key_info['x']), int(key_info['y'])
                kw, kh = int(key_info['width']), int(key_info['height'])
                if kx <= x <= kx + kw and ky <= y <= ky + kh:
                    return key_info['key']
        return None

    # ---------- improved pinch detection (pixel distance) ----------
    def is_pinch_gesture(self, landmarks, frame_shape):
        # Use Euclidean pixel distance between thumb tip and index tip
        if not landmarks:
            return False
        h, w = frame_shape[:2]
        thumb = landmarks[4]
        index = landmarks[8]
        dx = (thumb.x - index.x) * w
        dy = (thumb.y - index.y) * h
        pixel_dist = math.hypot(dx, dy)
        # threshold proportional to frame size: about 3% of diagonal
        diag = math.hypot(w, h)
        threshold = diag * 0.035  # tweakable
        return pixel_dist < threshold

    def process_key(self, key):
        if key.isalpha():
            if self.caps_lock:
                return key.upper()
            return key.lower()
        return key

    # ---------- handle key press: update stats + typing history ----------
    def handle_key_press(self, key):
        current_time = time.time()
        if current_time - self.last_click_time < self.click_delay:
            return
        self.last_click_time = current_time

        # increment keys_pressed for any real key action
        self.stats['keys_pressed'] += 1
        # only letters & space/backspace count as chars for WPM
        self.typing_history.append(current_time)

        # SPACE:
        if key == 'SPACE':
            if self.search_mode:
                self.search_text += ' '
            self.typed_text += ' '
            pyautogui.press('space')

        # BACKSPACE
        elif key == 'BACKSPACE':
            if self.search_mode and self.search_text:
                self.search_text = self.search_text[:-1]
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
            pyautogui.press('backspace')

        # special search hotkeys
        elif key in ['GOOGLE', 'INSTAGRAM', 'YOUTUBE']:
            if self.search_text.strip():
                search_url = self.search_urls[key] + self.search_text.replace(' ', '+')
                webbrowser.open(search_url)
                self.stats['hotkeys_used'] += 1
                self.search_text = ''
                self.search_mode = None
                self.typed_text = ''
            else:
                # enter search mode for selected engine
                self.search_mode = key
                self.search_text = ''

        # toggle caps lock if implemented key exists (not in current layout)
        elif key == 'CAPS':
            self.caps_lock = not self.caps_lock

        # normal characters
        else:
            processed_key = self.process_key(key)
            if self.search_mode:
                self.search_text += processed_key
            self.typed_text += processed_key
            # pyautogui write
            pyautogui.write(processed_key)

        # update wpm after handling
        self.calculate_wpm()

    def run(self):
        print("🚀 AI Virtual Keyboard - Updated Edition")
        print("=" * 50)
        print("👆 Point and pinch to type. Press 'q' to quit.")
        print("=" * 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            hover_key = None
            finger_pos = None
            pressed_key = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(100, 255, 150), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=2))
                    landmarks = hand_landmarks.landmark
                    finger_pos = self.get_finger_tip(landmarks, frame.shape)
                    self.draw_hand_skeleton(frame, landmarks, frame.shape)
                    if finger_pos:
                        cv2.circle(frame, finger_pos, 8, (100, 255, 150), -1)
                        cv2.circle(frame, finger_pos, 12, (150, 255, 200), 2)
                        hover_key = self.get_hovered_key(finger_pos)
                        # improved pinch check using frame size
                        if self.is_pinch_gesture(landmarks, frame.shape) and hover_key:
                            # start gesture timer when a new hovered key is pinched
                            if not self.is_gesturing or self.current_gesture_key != hover_key:
                                self.gesture_start_time = time.time()
                                self.is_gesturing = True
                                self.current_gesture_key = hover_key
                            # if held long enough, register press
                            if time.time() - self.gesture_start_time >= self.gesture_hold_time:
                                self.handle_key_press(hover_key)
                                pressed_key = hover_key
                                # after processing, reset gesture so accidental repeats are moderated
                                self.is_gesturing = False
                                self.current_gesture_key = None
                        else:
                            # reset if finger moves away
                            self.is_gesturing = False
                            self.current_gesture_key = None

            # draw UI
            self.draw_enhanced_keyboard(frame, hover_key, pressed_key)
            self.draw_typed_text_display(frame)
            if finger_pos and hover_key:
                # draw visual progress only when we are in the middle of a gesture
                if self.is_gesturing:
                    self.draw_gesture_feedback(frame, finger_pos, hover_key)

            # small overlay for stats (wpm)
            cv2.putText(frame, f"WPM: {self.stats['wpm']:.1f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

            cv2.imshow('AI Virtual Keyboard - Enhanced', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # session summary
        duration = int(time.time() - self.stats['session_start'])
        print("\n🎉 Session Complete!")
        print("=" * 40)
        print(f"📝 Keys pressed: {self.stats['keys_pressed']}")
        print(f"🔍 Searches made: {self.stats['hotkeys_used']}")
        print(f"⚡ Average WPM: {self.stats['wpm']:.1f}")
        print(f"⏱️ Duration: {duration} seconds")
        print("=" * 40)
        print("Thank you for using AI Virtual Keyboard! 👋")


if __name__ == "__main__":
    keyboard = VirtualKeyboard()
    keyboard.run()

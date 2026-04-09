import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import math

class JarvisGUI:
    def __init__(self, root, submit_callback=None):
        self.root = root
        self.root.title("JARVIS Assistant Interface")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")
        
        self.submit_callback = submit_callback
        
        self.setup_ui()
        
        # Animation variables
        self.orb_radius = 60
        self.orb_angle = 0.0
        self.is_processing = False
        self.animate_orb()
        
    def setup_ui(self):
        # Top frame: Persona indicator
        top_frame = tk.Frame(self.root, bg="#1e1e1e")
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.persona_label = tk.Label(
            top_frame, text="Persona: Default (Online)", 
            bg="#1e1e1e", fg="#00ffcc", font=("Consolas", 12, "bold")
        )
        self.persona_label.pack()
        
        # Middle frame: Orb and Chat Log
        mid_frame = tk.Frame(self.root, bg="#1e1e1e")
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20)
        
        # Left: Animated Orb (AI core representation)
        self.canvas = tk.Canvas(mid_frame, width=220, height=220, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Creating layered circles for a glowing orb effect
        self.orb_outer = self.canvas.create_oval(30, 30, 190, 190, fill="", outline="#004488", width=4)
        self.orb = self.canvas.create_oval(50, 50, 170, 170, fill="#0088ff", outline="#00ffff", width=2)
        self.orb_inner = self.canvas.create_oval(90, 90, 130, 130, fill="#ccffff", outline="")
        
        # Right: Conversation Log
        self.chat_log = scrolledtext.ScrolledText(
            mid_frame, bg="#252526", fg="#d4d4d4", font=("Consolas", 11), wrap=tk.WORD,
            insertbackground="white", highlightthickness=1, highlightbackground="#333333"
        )
        self.chat_log.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.chat_log.config(state=tk.DISABLED)
        
        # Bottom frame: Input Box
        bottom_frame = tk.Frame(self.root, bg="#1e1e1e")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=20)
        
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            bottom_frame, textvariable=self.input_var, bg="#3c3c3c", fg="#ffffff", 
            font=("Consolas", 12), insertbackground="white", relief=tk.FLAT
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.input_entry.bind("<Return>", self.on_submit)
        
        self.submit_btn = tk.Button(
            bottom_frame, text="Send", bg="#007acc", fg="white", 
            font=("Consolas", 11, "bold"), command=self.on_submit, relief=tk.FLAT, activebackground="#005f9e"
        )
        self.submit_btn.pack(side=tk.RIGHT, ipadx=15, ipady=4)
        
    def animate_orb(self):
        # Pulsing animation using sine wave
        self.orb_angle += 0.15 if not self.is_processing else 0.4
        pulse = math.sin(self.orb_angle) * (5 if not self.is_processing else 15)
        
        # Update main orb radius
        cx, cy = 110, 110
        r = self.orb_radius + pulse
        self.canvas.coords(self.orb, cx - r, cy - r, cx + r, cy + r)
        
        # Update colors based on processing state
        if self.is_processing:
            color = f"#ff{int(100 + pulse*5):02x}00" # Orange/Red pulse
            outline = "#ffcc00"
        else:
            color = f"#00{int(120 + pulse*5):02x}ff" # Blue cyan pulse
            outline = "#00ffff"
            
        self.canvas.itemconfig(self.orb, fill=color, outline=outline)
        
        self.root.after(40, self.animate_orb)
        
    def set_processing_state(self, is_processing):
        self.is_processing = is_processing
        
    def on_submit(self, event=None):
        user_text = self.input_var.get().strip()
        if not user_text:
            return
            
        self.input_var.set("")
        self.append_log("You", user_text)
        
        if self.submit_callback:
            self.set_processing_state(True)
            self.submit_btn.config(state=tk.DISABLED)
            
            # Run callback in a separate thread to prevent UI freezing
            threading.Thread(
                target=self._run_callback_thread, 
                args=(user_text,), 
                daemon=True
            ).start()
            
    def _run_callback_thread(self, text):
        try:
            # Assuming submit_callback returns a response string
            response = self.submit_callback(text)
            if response:
                self.append_log("JARVIS", response)
        except Exception as e:
            self.append_log("System", f"Error: {str(e)}")
        finally:
            self.root.after(0, self._restore_ui_state)

    def _restore_ui_state(self):
        self.set_processing_state(False)
        self.submit_btn.config(state=tk.NORMAL)
            
    def append_log(self, sender, text):
        # Thread-safe GUI update
        self.root.after(0, self._append_log_internal, sender, text)
        
    def _append_log_internal(self, sender, text):
        self.chat_log.config(state=tk.NORMAL)
        
        tag = sender.lower()
        self.chat_log.tag_config("you", foreground="#00ffff", font=("Consolas", 11, "bold"))
        self.chat_log.tag_config("jarvis", foreground="#ffaa00", font=("Consolas", 11, "bold"))
        self.chat_log.tag_config("system", foreground="#ff5555", font=("Consolas", 11, "bold"))
        
        self.chat_log.insert(tk.END, f"[{sender}] ", tag)
        self.chat_log.insert(tk.END, f"{text}\n\n")
        self.chat_log.see(tk.END)
        self.chat_log.config(state=tk.DISABLED)

    def update_persona(self, persona_name):
        self.root.after(0, lambda: self.persona_label.config(text=f"Persona: {persona_name}"))

def start_gui(submit_callback=None):
    root = tk.Tk()
    app = JarvisGUI(root, submit_callback)
    root.mainloop()

if __name__ == "__main__":
    # Test stub out
    def dummy_callback(text):
        time.sleep(1.5) # Simulate processing without freezing UI
        return f"I received your message: '{text}'. How can I help further?"

    start_gui(dummy_callback)

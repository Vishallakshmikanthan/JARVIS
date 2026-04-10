import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
import math
import psutil

class JarvisGUI:
    def __init__(self, root, submit_callback=None):
        self.root = root
        self.root.title("IronMan AI HUD")
        self.root.geometry("1000x700")
        self.root.configure(bg="#000b18")
        
        self.submit_callback = submit_callback
        
        # UI Colours
        self.c_bg = "#000b18"
        self.c_panel = "#001830"
        self.c_neon = "#00ffff"
        self.c_warn = "#ff3300"
        self.c_text = "#d4d4d4"
        self.c_dim = "#004488"
        
        self.setup_ui()
        
        # Animation variables
        self.orb_radius = 60
        self.orb_angle = 0.0
        self.is_processing = False
        
        # State variables
        self.active_skills = []
        self.current_agent = "Standby"
        
        self.animate_orb()
        self.update_system_stats()
        
    def setup_ui(self):
        # Configure grid for main layout
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Top frame: Persona indicator
        top_frame = tk.Frame(self.root, bg=self.c_bg)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=20)
        
        self.persona_label = tk.Label(
            top_frame, text="SYSTEM ONLINE: JARVIS", 
            bg=self.c_bg, fg=self.c_neon, font=("Consolas", 14, "bold"),
            relief=tk.FLAT
        )
        self.persona_label.pack(side=tk.LEFT)
        
        self.agent_label = tk.Label(
            top_frame, text="AGENT: Standby", 
            bg=self.c_bg, fg="#ffaa00", font=("Consolas", 14, "bold"),
            relief=tk.FLAT
        )
        self.agent_label.pack(side=tk.RIGHT)
        
        # Left Panel: HUD Stats & Orb
        left_panel = tk.Frame(self.root, bg=self.c_panel, highlightbackground=self.c_neon, highlightthickness=1)
        left_panel.grid(row=1, column=0, sticky="ns", pady=20, padx=(20, 10), ipadx=10, ipady=10)
        
        # Animated Orb (AI core representation)
        self.canvas = tk.Canvas(left_panel, width=220, height=220, bg=self.c_panel, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, pady=10)
        
        # Creating layered circles for a glowing orb effect
        self.orb_outer = self.canvas.create_oval(30, 30, 190, 190, fill="", outline=self.c_dim, width=4, dash=(4, 4))
        self.orb = self.canvas.create_oval(50, 50, 170, 170, fill="#0088ff", outline=self.c_neon, width=2)
        self.orb_inner = self.canvas.create_oval(90, 90, 130, 130, fill="#ccffff", outline="")
        
        # System Stats Frame
        stats_frame = tk.Frame(left_panel, bg=self.c_panel)
        stats_frame.pack(side=tk.TOP, fill=tk.X, pady=20, padx=10)
        
        # CPU Usage
        tk.Label(stats_frame, text="CPU USAGE", bg=self.c_panel, fg=self.c_neon, font=("Consolas", 10, "bold")).pack(anchor="w")
        
        cpu_frame = tk.Frame(stats_frame, bg=self.c_bg)
        cpu_frame.pack(fill=tk.X, pady=(2, 10))
        self.cpu_bar = tk.Canvas(cpu_frame, height=15, bg=self.c_bg, highlightthickness=1, highlightbackground=self.c_dim)
        self.cpu_bar.pack(fill=tk.X)
        self.cpu_fill = self.cpu_bar.create_rectangle(0, 0, 0, 15, fill=self.c_neon, outline="")
        self.cpu_text = self.cpu_bar.create_text(5, 8, text="0%", fill="white", anchor="w", font=("Consolas", 8))
        
        # RAM Usage
        tk.Label(stats_frame, text="MEM USAGE", bg=self.c_panel, fg=self.c_neon, font=("Consolas", 10, "bold")).pack(anchor="w")
        
        ram_frame = tk.Frame(stats_frame, bg=self.c_bg)
        ram_frame.pack(fill=tk.X, pady=(2, 10))
        self.ram_bar = tk.Canvas(ram_frame, height=15, bg=self.c_bg, highlightthickness=1, highlightbackground=self.c_dim)
        self.ram_bar.pack(fill=tk.X)
        self.ram_fill = self.ram_bar.create_rectangle(0, 0, 0, 15, fill=self.c_neon, outline="")
        self.ram_text = self.ram_bar.create_text(5, 8, text="0%", fill="white", anchor="w", font=("Consolas", 8))
        
        # Active Skills Array
        tk.Label(stats_frame, text="ACTIVE SKILLS", bg=self.c_panel, fg=self.c_neon, font=("Consolas", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.skills_label = tk.Label(
            stats_frame, text="None", 
            bg=self.c_panel, fg=self.c_text, font=("Consolas", 10), justify=tk.LEFT
        )
        self.skills_label.pack(anchor="w")
        
        # Right Panel: Chat Log & Input
        right_panel = tk.Frame(self.root, bg=self.c_bg)
        right_panel.grid(row=1, column=1, sticky="nsew", pady=20, padx=(10, 20))
        
        # Conversation Log
        self.chat_log = scrolledtext.ScrolledText(
            right_panel, bg=self.c_panel, fg=self.c_text, font=("Consolas", 11), wrap=tk.WORD,
            insertbackground=self.c_neon, highlightthickness=1, highlightbackground=self.c_dim
        )
        self.chat_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat_log.config(state=tk.DISABLED)
        
        # Bottom frame: Input Box
        bottom_frame = tk.Frame(right_panel, bg=self.c_bg)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            bottom_frame, textvariable=self.input_var, bg=self.c_panel, fg=self.c_neon, 
            font=("Consolas", 12, "bold"), insertbackground=self.c_neon, relief=tk.FLAT,
            highlightthickness=1, highlightbackground=self.c_dim
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.input_entry.bind("<Return>", self.on_submit)
        
        self.submit_btn = tk.Button(
            bottom_frame, text="EXECUTE", bg=self.c_dim, fg=self.c_neon, 
            font=("Consolas", 11, "bold"), command=self.on_submit, relief=tk.FLAT, 
            activebackground=self.c_neon, activeforeground="black"
        )
        self.submit_btn.pack(side=tk.RIGHT, ipadx=15, ipady=4)
        
        # HUD decorations
        deco_frame = tk.Frame(self.root, bg=self.c_bg, height=5)
        deco_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))
        tk.Frame(deco_frame, bg=self.c_neon, height=2, width=150).pack(side=tk.LEFT)
        tk.Frame(deco_frame, bg=self.c_neon, height=2, width=150).pack(side=tk.RIGHT)

        
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
        
    def update_system_stats(self):
        """Update CPU, RAM stats and Agent active state in the HUD"""
        try:
            # Requires psutil
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            # Update bar filled rectangles based on % Width max ~190
            max_w = 190 
            
            self.cpu_bar.coords(self.cpu_fill, 0, 0, max_w * (cpu/100.0), 15)
            self.cpu_bar.itemconfig(self.cpu_text, text=f"{cpu}%")
            if cpu > 80:
                self.cpu_bar.itemconfig(self.cpu_fill, fill=self.c_warn)
                self.cpu_bar.itemconfig(self.cpu_text, fill="black")
            else:
                self.cpu_bar.itemconfig(self.cpu_fill, fill=self.c_neon)
                self.cpu_bar.itemconfig(self.cpu_text, fill="black")
                
            self.ram_bar.coords(self.ram_fill, 0, 0, max_w * (ram/100.0), 15)
            self.ram_bar.itemconfig(self.ram_text, text=f"{ram}%")
            if ram > 85:
                self.ram_bar.itemconfig(self.ram_fill, fill=self.c_warn)
                self.ram_bar.itemconfig(self.ram_text, fill="black")
            else:
                self.ram_bar.itemconfig(self.ram_fill, fill=self.c_neon)
                self.ram_bar.itemconfig(self.ram_text, fill="black")
                
            # Update Active Agent Label
            self.agent_label.config(text=f"AGENT: {self.current_agent}")
            
            # Update Skills Array Label
            if not self.active_skills:
                self.skills_label.config(text="- NULL", fg=self.c_dim)
            else:
                txt = "\n".join([f"> {s}" for s in set(self.active_skills)])
                self.skills_label.config(text=txt, fg=self.c_neon)
                
        except Exception as e:
            pass
            
        self.root.after(1500, self.update_system_stats)
        
    def set_processing_state(self, is_processing):
        self.is_processing = is_processing
        
    def set_agent(self, agent_name):
        self.current_agent = agent_name
        
    def set_active_skills(self, skills_list):
        self.active_skills = skills_list
        
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

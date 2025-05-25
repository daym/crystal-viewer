#!/usr/bin/env python3

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.GLU import gluErrorString # For error checking
import numpy as np
import glm # PyGLM for matrix math
import ctypes

# --- GL Error Checking Utility ---
def check_gl_error(context_message=""):
    err = glGetError()
    if err != GL_NO_ERROR:
        try:
            error_str = gluErrorString(err)
            if isinstance(error_str, bytes): error_str = error_str.decode()
        except Exception:
            error_str = f"OpenGL Error Code: {err}"
        print(f"OpenGL Error after {context_message}: {error_str}")
        # import traceback
        # traceback.print_stack() # Print stack to see where error was checked from
        # raise RuntimeError(f"OpenGL Error after {context_message}: {error_str}")

# --- Shader Definitions ---
VERTEX_SHADER_LINES = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aLineCoord; 
uniform mat4 model; uniform mat4 view; uniform mat4 projection;
out float v_LineCoord;
void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); v_LineCoord = aLineCoord; }
"""
FRAGMENT_SHADER_LINES = """
#version 330 core
out vec4 FragColor; uniform vec4 lineColor; uniform bool isDashed;
uniform float dashSegmentLength = 0.1; 
in float v_LineCoord;
void main() { if (isDashed && mod(v_LineCoord, dashSegmentLength*2.0) > dashSegmentLength) discard; FragColor = lineColor; }
"""
VERTEX_SHADER_POINTS = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model; uniform mat4 view; uniform mat4 projection; uniform float pointRenderSize;
void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); gl_PointSize = pointRenderSize; }
"""
FRAGMENT_SHADER_POINTS = """
#version 330 core
out vec4 FragColor; uniform vec4 pointColor;
void main() { if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard; FragColor = pointColor; }
"""

# --- Crystallographic Data & Helper Functions (Same as before) ---
def get_cartesian_vectors(a,b,c,alpha_deg,beta_deg,gamma_deg):
    alpha,beta,gamma = np.radians(alpha_deg),np.radians(beta_deg),np.radians(gamma_deg)
    v_a = np.array([a,0,0],dtype=np.float32)
    v_b_x,v_b_y,v_b_z = b*np.cos(gamma),b*np.sin(gamma),0; v_b = np.array([v_b_x,v_b_y,v_b_z],dtype=np.float32)
    v_c_x = c*np.cos(beta)
    sin_gamma_safe = np.sin(gamma); sin_gamma_safe = 1e-9*(1 if sin_gamma_safe>=0 else -1) if abs(sin_gamma_safe)<1e-9 else sin_gamma_safe
    v_c_y = c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/sin_gamma_safe
    val_sqrt = 1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    v_c_z = c*np.sqrt(max(0,val_sqrt))/sin_gamma_safe; v_c = np.array([v_c_x,v_c_y,v_c_z],dtype=np.float32)
    if alpha_deg==90 and beta_deg==90 and gamma_deg==90: v_a,v_b,v_c = np.array([a,0,0],dtype=np.float32),np.array([0,b,0],dtype=np.float32),np.array([0,0,c],dtype=np.float32)
    elif alpha_deg==90 and beta_deg==90 and gamma_deg==120: v_a,v_c = np.array([a,0,0],dtype=np.float32),np.array([0,0,c],dtype=np.float32); v_b = np.array([b*np.cos(np.radians(120)),b*np.sin(np.radians(120)),0],dtype=np.float32)
    return v_a,v_b,v_c
def get_cell_vertices_from_vectors(v_a,v_b,v_c,origin=np.array([0,0,0],dtype=np.float32)):
    o=np.array(origin,dtype=np.float32); return [o,o+v_a,o+v_b,o+v_c,o+v_a+v_b,o+v_a+v_c,o+v_b+v_c,o+v_a+v_b+v_c]
def corners_to_line_vertex_data(corners):
    if not corners or len(corners)!=8: return np.array([],dtype=np.float32)
    edges=[0,1,0,2,0,3,1,4,1,5,2,4,2,6,3,5,3,6,4,7,5,7,6,7]; v_data=[]
    for i in range(0,len(edges),2): p1,p2=corners[edges[i]],corners[edges[i+1]]; v_data.extend(p1);v_data.append(0.0); v_data.extend(p2);v_data.append(1.0)
    return np.array(v_data,dtype=np.float32)
SPACE_GROUP_DATA = {
    "P1":{"conventional":{"params":(6,7,8,70,80,85),"color":(1,0,0,1)},"primitive_is_conventional":True,"primitive_color":(0,1,0,1),"lattice_points_conv_frac":[[0,0,0]]},
    "Pm-3m":{"conventional":{"params":(5,5,5,90,90,90),"color":(1,0,0,1)},"primitive_is_conventional":True,"primitive_color":(0,1,0,1),"lattice_points_conv_frac":[[0,0,0]]},
    "Fm-3m":{"conventional":{"params":(5,5,5,90,90,90),"color":(1,0,0,1)},"primitive_vectors_from_conv_abc":[lambda a,b,c:np.array([a/2,b/2,0],dtype=np.float32),lambda a,b,c:np.array([a/2,0,c/2],dtype=np.float32),lambda a,b,c:np.array([0,b/2,c/2],dtype=np.float32)],"primitive_color":(0,0,1,1),"lattice_points_conv_frac":[[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]},
    "Amm2":{"conventional":{"params":(3.5,4.5,6.0,90,90,90),"color":(1,0,0,1)},"primitive_vectors_from_conv_abc":[lambda a,b,c:np.array([a,0,0],dtype=np.float32),lambda a,b,c:np.array([0,b/2,c/2],dtype=np.float32),lambda a,b,c:np.array([0,-b/2,c/2],dtype=np.float32)],"primitive_color":(0,0,1,1),"lattice_points_conv_frac":[[0,0,0],[0,0.5,0.5]]},
    "R3:H":{"conventional":{"params":(5,5,17,90,90,120),"color":(1,0,0,1)},"primitive_params":(6.3,6.3,6.3,46,46,46),"primitive_color":(0,0,1,1),"lattice_points_conv_frac":[[0,0,0],[2/3,1/3,1/3],[1/3,2/3,2/3]]}
}
SPACE_GROUP_KEYS = list(SPACE_GROUP_DATA.keys())

class CrystalGLArea(Gtk.GLArea):
    def __init__(self):
        super().__init__(); self.set_has_depth_buffer(True); self.set_has_alpha(True); self.set_required_version(3,3)
        self.connect("realize",self.on_realize); self.connect("unrealize",self.on_unrealize)
        self.connect("render",self.on_render); self.connect("resize",self.on_resize)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK|Gdk.EventMask.BUTTON_RELEASE_MASK|Gdk.EventMask.POINTER_MOTION_MASK|Gdk.EventMask.SCROLL_MASK)
        self.connect("button-press-event",self.on_mouse_press); self.connect("button-release-event",self.on_mouse_release)
        self.connect("motion-notify-event",self.on_mouse_motion); self.connect("scroll-event",self.on_scroll)
        self.rotation_x,self.rotation_y=30.0,-45.0; self.distance,self.pan_x,self.pan_y=25.0,0.0,0.0
        self.last_mouse_pos=None; self.mouse_left_down,self.mouse_middle_down=False,False
        self.line_shader,self.point_shader=None,None; self.line_vao,self.line_vbo=None,None; self.point_vao,self.point_vbo=None,None
        self.conv_cell_line_data,self.prim_cell_line_data,self.seam_line_data,self.lattice_point_data = [np.array([],dtype=np.float32)]*4
        self._all_line_data_to_upload,self._point_data_to_upload = np.array([],dtype=np.float32), np.array([],dtype=np.float32)
        self._needs_vbo_upload=True; self.current_sg_name,self.current_sg_data_dict=None,None
        self.line_width_solid,self.line_width_dashed,self.point_render_size = 2.0,1.5,7.0

    def on_realize(self, area):
        area.make_current()
        try:
            print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}"); check_gl_error("Start of on_realize")
            
            print("Compiling shaders...")
            self.line_shader = compileProgram(compileShader(VERTEX_SHADER_LINES,GL_VERTEX_SHADER),compileShader(FRAGMENT_SHADER_LINES,GL_FRAGMENT_SHADER)); check_gl_error("Compile line_shader")
            self.point_shader = compileProgram(compileShader(VERTEX_SHADER_POINTS,GL_VERTEX_SHADER),compileShader(FRAGMENT_SHADER_POINTS,GL_FRAGMENT_SHADER)); check_gl_error("Compile point_shader")
            print("Shaders compiled.")

            print("Generating VAOs/VBOs (IDs only)...")
            self.line_vao = glGenVertexArrays(1)
            self.line_vbo = glGenBuffers(1)
            check_gl_error("Generated line VAO/VBO IDs")

            self.point_vao = glGenVertexArrays(1)
            self.point_vbo = glGenBuffers(1)
            check_gl_error("Generated point VAO/VBO IDs")
            print("VAOs/VBOs IDs generated.")

            print("Setting up Line VAO attributes...")
            glBindVertexArray(self.line_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo) # Bind VBO even if no data yet
            # Initialize buffer with some size, but no data yet, or empty data
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW) # IMPORTANT: Initialize buffer store
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,4*sizeof(GLfloat),ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
            glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,4*sizeof(GLfloat),ctypes.c_void_p(3*sizeof(GLfloat))); glEnableVertexAttribArray(1); check_gl_error("Line VAO attribute setup")
            print("Line VAO attributes set.")

            print("Setting up Point VAO attributes...")
            glBindVertexArray(self.point_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo) # Bind VBO
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW) # IMPORTANT: Initialize buffer store
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(GLfloat),ctypes.c_void_p(0)); glEnableVertexAttribArray(0); check_gl_error("Point VAO attribute setup")
            print("Point VAO attributes set.")

            glBindBuffer(GL_ARRAY_BUFFER,0); glBindVertexArray(0)
            
            glEnable(GL_DEPTH_TEST);glEnable(GL_PROGRAM_POINT_SIZE);glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);glClearColor(0.15,0.15,0.25,1.0); check_gl_error("GL enables")
            
            # DO NOT CALL _upload_vbo_data() here yet
            # if self._needs_vbo_upload: self._upload_vbo_data() 
            print("on_realize finished successfully.")

        except Exception as e: print(f"Error during GL realization: {e}"); import traceback; traceback.print_exc()

    def on_unrealize(self, area):
        area.make_current()
        if self.line_shader: glDeleteProgram(self.line_shader); self.line_shader=None
        if self.point_shader: glDeleteProgram(self.point_shader); self.point_shader=None
        if self.line_vao is not None: glDeleteVertexArrays(1,np.array([self.line_vao],dtype=np.uint32)); self.line_vao=None
        if self.line_vbo is not None: glDeleteBuffers(1,np.array([self.line_vbo],dtype=np.uint32)); self.line_vbo=None
        if self.point_vao is not None: glDeleteVertexArrays(1,np.array([self.point_vao],dtype=np.uint32)); self.point_vao=None
        if self.point_vbo is not None: glDeleteBuffers(1,np.array([self.point_vbo],dtype=np.uint32)); self.point_vbo=None
        check_gl_error("on_unrealize")

    def on_resize(self, area, width, height): area.make_current();glViewport(0,0,width,height);self.queue_render();check_gl_error("on_resize")

    def on_render(self, area, context):
        area.make_current()
        if self._needs_vbo_upload: self._upload_vbo_data()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); check_gl_error("glClear in on_render")
        w,h=self.get_allocated_width(),self.get_allocated_height(); aspect=float(w)/(h if h>0 else 1)
        proj=glm.perspective(glm.radians(45.0),aspect,0.1,200.0)
        view=glm.translate(glm.mat4(1.0),glm.vec3(self.pan_x,self.pan_y,-self.distance))
        view=glm.rotate(view,glm.radians(self.rotation_x),glm.vec3(1,0,0)); view=glm.rotate(view,glm.radians(self.rotation_y),glm.vec3(0,1,0))
        model=glm.mat4(1.0)

        if self.line_shader and self.line_vao and self._all_line_data_to_upload.size>0:
            glUseProgram(self.line_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.line_shader,"projection"),1,GL_FALSE,glm.value_ptr(proj))
            glUniformMatrix4fv(glGetUniformLocation(self.line_shader,"view"),1,GL_FALSE,glm.value_ptr(view))
            glUniformMatrix4fv(glGetUniformLocation(self.line_shader,"model"),1,GL_FALSE,glm.value_ptr(model))
            glBindVertexArray(self.line_vao); check_gl_error("Use line_shader and bind line_vao")
            offset=0
            if self.conv_cell_line_data.size>0:
                glLineWidth(self.line_width_solid); glUniform4fv(glGetUniformLocation(self.line_shader,"lineColor"),1,self.current_sg_data_dict["conventional"]["color"]); glUniform1i(glGetUniformLocation(self.line_shader,"isDashed"),GL_FALSE)
                n=self.conv_cell_line_data.shape[0]//4; glDrawArrays(GL_LINES,offset,n); offset+=n; check_gl_error("Draw conv_cell lines")
            if self.prim_cell_line_data.size>0:
                glLineWidth(self.line_width_solid*1.1); glUniform4fv(glGetUniformLocation(self.line_shader,"lineColor"),1,self.current_sg_data_dict.get("primitive_color",(0,1,0,1))); glUniform1i(glGetUniformLocation(self.line_shader,"isDashed"),GL_FALSE)
                n=self.prim_cell_line_data.shape[0]//4; glDrawArrays(GL_LINES,offset,n); offset+=n; check_gl_error("Draw prim_cell lines")
            if self.seam_line_data.size>0:
                glLineWidth(self.line_width_dashed); s_col=list(self.current_sg_data_dict["conventional"]["color"]);s_col[3]=0.5; glUniform4fv(glGetUniformLocation(self.line_shader,"lineColor"),1,s_col); glUniform1i(glGetUniformLocation(self.line_shader,"isDashed"),GL_TRUE)
                n=self.seam_line_data.shape[0]//4; glDrawArrays(GL_LINES,offset,n); check_gl_error("Draw seam lines")
            glBindVertexArray(0)

        if self.point_shader and self.point_vao and self._point_data_to_upload.size>0:
            glUseProgram(self.point_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.point_shader,"projection"),1,GL_FALSE,glm.value_ptr(proj))
            glUniformMatrix4fv(glGetUniformLocation(self.point_shader,"view"),1,GL_FALSE,glm.value_ptr(view))
            glUniformMatrix4fv(glGetUniformLocation(self.point_shader,"model"),1,GL_FALSE,glm.value_ptr(model))
            glUniform4fv(glGetUniformLocation(self.point_shader,"pointColor"),1,[0.8,0.8,0.2,1.0]); glUniform1f(glGetUniformLocation(self.point_shader,"pointRenderSize"),self.point_render_size)
            glBindVertexArray(self.point_vao); check_gl_error("Use point_shader and bind point_vao")
            glDrawArrays(GL_POINTS,0,self._point_data_to_upload.shape[0]//3); check_gl_error("Draw points")
            glBindVertexArray(0)
        glUseProgram(0); return True

    def _prepare_data_for_vbo(self):
        if not self.current_sg_data_dict:
            self.conv_cell_line_data,self.prim_cell_line_data,self.seam_line_data,self.lattice_point_data = [np.array([],dtype=np.float32)]*4
            self._all_line_data_to_upload,self._point_data_to_upload = np.array([],dtype=np.float32),np.array([],dtype=np.float32)
            self._needs_vbo_upload=True; return
        data = self.current_sg_data_dict; conv_params = data["conventional"]["params"]
        conv_va,conv_vb,conv_vc = get_cartesian_vectors(*conv_params)
        conv_corners = get_cell_vertices_from_vectors(conv_va,conv_vb,conv_vc)
        self.conv_cell_line_data = corners_to_line_vertex_data(conv_corners)
        self.prim_cell_line_data = np.array([],dtype=np.float32); p_orig=np.array(data.get("primitive_offset_from_conv_origin",[0,0,0]),dtype=np.float32); p_corn=None
        if data.get("primitive_is_conventional",False):p_corn=get_cell_vertices_from_vectors(conv_va,conv_vb,conv_vc,origin=p_orig)
        elif "primitive_params" in data:pp=data["primitive_params"];p_va,p_vb,p_vc=get_cartesian_vectors(*pp);p_corn=get_cell_vertices_from_vectors(p_va,p_vb,p_vc,origin=p_orig)
        elif "primitive_vectors_from_conv_abc" in data:
            ca,cb,cc=conv_params[0:3];vf=data["primitive_vectors_from_conv_abc"];p_va,p_vb,p_vc=vf[0](ca,cb,cc),vf[1](ca,cb,cc),vf[2](ca,cb,cc)
            p_corn=get_cell_vertices_from_vectors(p_va,p_vb,p_vc,origin=p_orig)
        if p_corn:
            is_diff=not(data.get("primitive_is_conventional",False) and np.allclose(p_corn[0],conv_corners[0]) and np.allclose(p_corn[1],conv_corners[1]))
            if is_diff or not data.get("primitive_is_conventional",False): self.prim_cell_line_data=corners_to_line_vertex_data(p_corn)
        seam_v_list=[]
        for v_off in [conv_va,conv_vb,conv_vc]: n_o=conv_corners[0]+v_off;n_c=get_cell_vertices_from_vectors(conv_va,conv_vb,conv_vc,origin=n_o);seam_v_list.append(corners_to_line_vertex_data(n_c))
        self.seam_line_data=np.concatenate(seam_v_list) if seam_v_list else np.array([],dtype=np.float32)
        pt_list=[]
        for frac_pt in data.get("lattice_points_conv_frac",[[0,0,0]]): pt_list.extend(conv_va*frac_pt[0]+conv_vb*frac_pt[1]+conv_vc*frac_pt[2])
        self.lattice_point_data=np.array(pt_list,dtype=np.float32)
        _lines_upload=[d for d in [self.conv_cell_line_data,self.prim_cell_line_data,self.seam_line_data] if d.size>0]
        self._all_line_data_to_upload=np.concatenate(_lines_upload) if _lines_upload else np.array([],dtype=np.float32)
        self._point_data_to_upload=self.lattice_point_data; self._needs_vbo_upload=True

    def _upload_vbo_data(self):
        if not self.get_realized() or self.line_vbo is None or self.point_vbo is None:
            self._needs_vbo_upload = True; return
        self.make_current()
        print("Binding buffer...")
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        print("Uploading line data...")
        if self._all_line_data_to_upload.size > 0:
            # Ensure C-contiguous
            line_data_contiguous = np.ascontiguousarray(self._all_line_data_to_upload, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, line_data_contiguous.nbytes, line_data_contiguous, GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW) # Clear
        check_gl_error("Upload line data")

        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        if self._point_data_to_upload.size > 0:
            # Ensure C-contiguous
            point_data_contiguous = np.ascontiguousarray(self._point_data_to_upload, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, point_data_contiguous.nbytes, point_data_contiguous, GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW) # Clear
        check_gl_error("Upload point data")
        
        print("Unbinding buffer...")
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        print("end...")
        self._needs_vbo_upload = False

    def update_crystal_data(self, sg_name):
        self.current_sg_name=sg_name; self.current_sg_data_dict=SPACE_GROUP_DATA.get(sg_name)
        self._prepare_data_for_vbo()
        if self.get_realized() and self.line_vbo is not None: self._upload_vbo_data()
        self.queue_render()

    def on_mouse_press(self,w,e):self.last_mouse_pos=(e.x,e.y);self.mouse_left_down=(e.button==Gdk.BUTTON_PRIMARY);self.mouse_middle_down=(e.button==Gdk.BUTTON_MIDDLE or e.button==Gdk.BUTTON_SECONDARY)
    def on_mouse_release(self,w,e):
        if e.button==Gdk.BUTTON_PRIMARY:self.mouse_left_down=False
        elif e.button==Gdk.BUTTON_MIDDLE or e.button==Gdk.BUTTON_SECONDARY:self.mouse_middle_down=False
        self.last_mouse_pos=None
    def on_mouse_motion(self,w,e):
        if not self.last_mouse_pos:return
        dx,dy=e.x-self.last_mouse_pos[0],e.y-self.last_mouse_pos[1]
        if self.mouse_left_down:self.rotation_y+=dx*0.35;self.rotation_x+=dy*0.35
        elif self.mouse_middle_down:p_s=0.02*(self.distance/20.0);self.pan_x+=dx*p_s;self.pan_y-=dy*p_s
        self.last_mouse_pos=(e.x,e.y);self.queue_render()
    def on_scroll(self,w,e):
        _,d_y=e.get_scroll_deltas();z_f=1.5
        if d_y<0:self.distance-=z_f*abs(d_y)
        elif d_y>0:self.distance+=z_f*abs(d_y)
        self.distance=max(2.0,min(self.distance,150.0));self.queue_render()

class MainWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="GTK Crystal Viewer (Core GL)");self.set_default_size(850,700);self.connect("destroy",Gtk.main_quit)
        main_vb=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=6);self.add(main_vb)
        ctrl_hb=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=6);ctrl_hb.set_margin_top(10);ctrl_hb.set_margin_bottom(5);ctrl_hb.set_margin_start(10);ctrl_hb.set_margin_end(10)
        main_vb.pack_start(ctrl_hb,False,False,0);sg_lbl=Gtk.Label(label="Space Group:");ctrl_hb.pack_start(sg_lbl,False,False,0)
        self.sg_combo=Gtk.ComboBoxText();[self.sg_combo.append_text(k) for k in SPACE_GROUP_KEYS];self.sg_combo.set_active(0);ctrl_hb.pack_start(self.sg_combo,True,True,0)
        self.gl_area=CrystalGLArea();main_vb.pack_start(self.gl_area,True,True,0)
        self.sg_combo.connect("changed",self.on_sg_combo_changed)
        if self.sg_combo.get_active_text():self.gl_area.update_crystal_data(self.sg_combo.get_active_text())
    def on_sg_combo_changed(self,combo):
        active_text=combo.get_active_text()
        if active_text:self.gl_area.update_crystal_data(active_text)

def main():win=MainWindow();win.show_all();Gtk.main()
if __name__ == '__main__': main()


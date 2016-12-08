
def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
if not module_exists("glfw"):
    print "Warning : glfw doesn't exists..."
import random

class Environment:

    def __init__(self, jointIndex, getNewMeshCallback, WIDTH=1080, HEIGHT=680, isNormalize=True):
        if not glfw.init():
            return

        self.jointIndex = jointIndex
        self.getNewMeshCallback = getNewMeshCallback

        self.colorTable = []

        self.colorTable.append([200 /255.0, 33 /255.0, 93 /255.0])
        self.colorTable.append([42 /255.0, 159 /255.0, 188 /255.0])
        self.colorTable.append([247 /255.0, 147 /255.0, 29 /255.0])
        #self.colorTable.append([247 /255.0, 147 /255.0, 29 /255.0])
        #for joints in jointIndex:
        #    self.colorTable.append([random.random(), random.random(), random.random()])

        self.window = glfw.create_window(WIDTH, HEIGHT, "Mesh Loader", None, None)
        #self.window = glfw.create_window(500, 500, "aweg", None, None)

        self.isRunning = True
        self.isPlaying = True
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.ratio = self.WIDTH / float(self.HEIGHT)

        self.meshes = self.getNewMeshCallback(0)
        self.frameNo = 0

        if isNormalize:
            self.meshDistance = 2
            self.cameraPos=[2, 0, 50, 2, 0, 0, 0, 1, 0]
        else:
            self.meshDistance = 100
            self.cameraPos=[60, 0, 100, 60, 0, 0, 0, 1, 0]

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_window_size_callback(self.window, self.resize_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_SMOOTH)

    def __call__(self):

        #call this once
        self.resize_callback(self.window, self.WIDTH, self.HEIGHT)
        skip = 0

        if self.isPlaying:
            self.render_callback(self.meshes)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        return self.isRunning

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key==glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                self.isRunning = False
            if key==glfw.KEY_SPACE:
                self.frameNo += 1
                self.meshes = self.getNewMeshCallback(self.frameNo)
        #self.render_callback()

    def resize_callback(self, window, newWidth, newHeight):
        self.ratio = newWidth / float(newHeight)
        self.WIDTH = newWidth
        self.HEIGHT = newHeight
        glViewport(0, 0, self.WIDTH, self.HEIGHT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-10, 10, -10, 10, -10, 10)
        #gluPerspective(10.0, self.ratio, 0.1, 100.0)
        #gluLookAt(self.cameraPos[0],self.cameraPos[1],self.cameraPos[2],
        #self.cameraPos[3],self.cameraPos[4],self.cameraPos[5],
        #self.cameraPos[6],self.cameraPos[7],self.cameraPos[8])
        #gluLookat()

    def render_callback(self, meshes):
        glClearColor(1, 1, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)

        glLoadIdentity()
        glTranslatef(-2, 0, 0)

        #print meshes[0]
        #self.isRunning = False
        #return

        def drawBVHModel(mesh, offset_x):
            glLineWidth(2.0)
            glPushMatrix()
            glTranslatef(offset_x, 0, 0)
            #glRotatef(60, 1, 0, 0)
            #glRotatef(45, 0, 1, 0)
            
            for i in range(0, len(self.jointIndex)):
                if self.jointIndex[i] == -1:
                    continue

                pPos = mesh[self.jointIndex[i]]

                if i >0 and mesh[i][0] == 0 and mesh[i][1] == 0 and mesh[i][2] ==0:
                    continue
                if self.jointIndex[i] >0 and pPos[0]==0 and pPos[1]==0 and pPos[2]==0:
                    continue
                #glColor3d(self.colorTable[i][0],self.colorTable[i][1],self.colorTable[i][2])
                glLineWidth(4.0)
                glBegin(GL_LINES)
                glVertex3f(mesh[i][0], mesh[i][1], mesh[i][2])
                glVertex3f(pPos[0], pPos[1], pPos[2])
                glEnd()

            glPopMatrix()

        Xoffset = self.meshDistance * -1
        glScalef(2.0, 2.0, 2.0)
        i=0
        for mesh in meshes:        
            glColor3d(self.colorTable[i][0],self.colorTable[i][1],self.colorTable[i][2])
            Xoffset = Xoffset + self.meshDistance
            drawBVHModel(mesh, Xoffset)
            i=i+1

'''
def renderBone(x0, y0, z0, x1, y1, z1):
    dir_x = x1-x0
    dir_y = y1-y0
    dir_z = z1-z0
    bone_length = math.sqrtf(dir_x*dir_x+dir_y*dir_y+dir_z*dir_z)

    quad_obj = gluNewQuadric()
    gluQuadricDrawStyle( quad_obj, GLU_FILL )
    gluQuadricNormals( quad_obj, GLU_SMOOTH )

    glPushMatrix()
    glTranslatef( x0, y0, z0 )
'''

#!/usr/bin/python3

from threading import Thread
import rospy
import math
import numpy as np
import time
from geometry_msgs.msg import Vector3
from tobe3_real.tobe import Tobe


class WalkFunc:
    """
    Walk Joint Function CPG style
    Provides parameters for walking, given swing amplitude A, motion phase mu
    """

    def __init__(self):
        K1 = 0.35  # home position leg extension
        K2 = -0.15  # -0.1 home position leg roll angle
        K3 = -0.45  # -0.2 home position leg pitch angle
        K4 = 0.05  # home position foot roll angle
        K5 = -0.12  # home position foot pitch angle
        self.eta_R = K1
        self.eta_L = K1
        self.phi_Rleg = np.array([K2, K3, 0])
        self.phi_Lleg = np.array([-K2, K3, 0])
        self.phi_Rfoot = np.array([K4, K5])
        self.phi_Lfoot = np.array([K4, K5])
        self.A = np.array([0, 0, 0])
        self.phase = -math.pi
        self.At1 = 0

    def eta_leg_lift(self, A, phase, K_mu0, K_mu1, K_mu2):
        mu = phase
        K6 = 0.04   # ground push constant
        K7 = 0      # ground push intensifier
        K8 = 0.06   # 0.06 step height constant
        K9 = 0.03   # 0.03 step height intensifier
        if ((mu > K_mu0) and (mu < K_mu1)):  # during right swing phase
            if mu < K_mu2:  # during leg lifting
                eta_R = math.sin(0.5*math.pi*(mu-K_mu0) /
                                 (K_mu2-K_mu0))*(K8+K9*max(abs(A)))
            else:  # during leg lowering
                eta_R = math.sin(
                    0.5*math.pi*(1+(mu-K_mu2)/(K_mu1-K_mu2)))*(K8+K9*max(abs(A)))
        else:  # during right stance phase
            if mu >= K_mu1:
                nu_R1 = -math.pi+math.pi*(mu-K_mu1)/(2*math.pi+K_mu0-K_mu1)
            else:
                nu_R1 = -math.pi+math.pi * \
                    (mu-K_mu1+2*math.pi)/(2*math.pi+K_mu0-K_mu1)
            eta_R = math.sin(0.5*nu_R1)*(K6+K7*max(abs(A)))
        if ((mu > K_mu0-math.pi) and (mu < K_mu1-math.pi)):  # during left swing phase
            if mu < K_mu2-math.pi:  # during leg lifting
                eta_L = math.sin(0.5*math.pi*(mu-K_mu0+math.pi) /
                                 (K_mu2-K_mu0))*(K8+K9*max(abs(A)))
            else:  # during leg lowering
                eta_L = math.sin(
                    0.5*math.pi*(1+(mu-K_mu2+math.pi)/(K_mu1-K_mu2)))*(K8+K9*max(abs(A)))
        else:  # during right stance phase
            if mu >= K_mu1-math.pi:
                nu_L1 = -math.pi+math.pi * \
                    (mu-K_mu1+math.pi)/(2*math.pi+K_mu0-K_mu1)
            else:
                nu_L1 = -math.pi+math.pi * \
                    (mu-K_mu1+3*math.pi)/(2*math.pi+K_mu0-K_mu1)
            eta_L = math.sin(0.5*nu_L1)*(K6+K7*max(abs(A)))
        return eta_R, eta_L

    def hip_ankle(self, phase, K_mu0, K_mu1):  # hip swing + ankle pronation/supination
        mu = phase
        # stance hip position (from home) just before opposite leg swings
        prestance = 0.1
        stmax = -0.05     # max stance hip displacement during opposite leg swing
        preswing = 0.1    # swing hip position (from home) just before swing
        swmax = 0		# max swing hip displacement during swing
        prepro = -0.2     # stance ankle angle just before pronation sequence
        presup = 0.1  # swing ankle angle just before supination sequence
        pron = 0		# max angle change during pronation of non-swing foot
        sup = 0		# max angle change during swing phase supination
        if ((mu >= K_mu0-math.pi) and (mu < K_mu1-math.pi)):  # left swing phase
            hip_R = prestance+stmax * \
                math.sin(math.pi*(mu-K_mu0+math.pi)/(K_mu1-K_mu0))
            hip_L = preswing+swmax * \
                math.sin(math.pi*(mu-K_mu0+math.pi)/(K_mu1-K_mu0))
            ank_R = prepro-pron * \
                math.sin(math.pi*(mu-K_mu0+math.pi)/(K_mu1-K_mu0))
            ank_L = presup+sup * \
                math.sin(math.pi*(mu-K_mu0+math.pi)/(K_mu1-K_mu0))
        else:
            if ((mu >= K_mu1-math.pi) and (mu < K_mu0)):  # double support leading to right swing
                hip_R = prestance+(-preswing-prestance)*math.sin(0.5 *
                                                                 math.pi*(mu-K_mu1+math.pi)/(K_mu0-K_mu1+math.pi))
                hip_L = preswing+(-prestance-preswing)*math.sin(0.5 *
                                                                math.pi*(mu-K_mu1+math.pi)/(K_mu0-K_mu1+math.pi))
                ank_R = prepro+(presup-prepro) * \
                    (mu-K_mu1+math.pi)/(K_mu0-K_mu1+math.pi)
                ank_L = presup+(prepro-presup) * \
                    (mu-K_mu1+math.pi)/(K_mu0-K_mu1+math.pi)
            else:
                if ((mu >= K_mu0) and (mu < K_mu1)):  # right swing phase
                    hip_R = -preswing-swmax * \
                        math.sin(math.pi*(mu-K_mu0)/(K_mu1-K_mu0))
                    hip_L = -prestance-stmax * \
                        math.sin(math.pi*(mu-K_mu0)/(K_mu1-K_mu0))
                    ank_R = presup+sup * \
                        math.sin(math.pi*(mu-K_mu0)/(K_mu1-K_mu0))
                    ank_L = prepro-pron * \
                        math.sin(math.pi*(mu-K_mu0)/(K_mu1-K_mu0))
                else:
                    if mu >= K_mu1:  # double support leading to left swing, part 1
                        hip_R = -preswing + \
                            (preswing+prestance)*math.sin(0.5 *
                                                          math.pi*(mu-K_mu1)/(K_mu0-K_mu1+math.pi))
                        hip_L = -prestance + \
                            (prestance+preswing)*math.sin(0.5 *
                                                          math.pi*(mu-K_mu1)/(K_mu0-K_mu1+math.pi))
                        ank_R = presup+(prepro-presup) * \
                            (mu-K_mu1)/(K_mu0-K_mu1+math.pi)
                        ank_L = prepro+(presup-prepro) * \
                            (mu-K_mu1)/(K_mu0-K_mu1+math.pi)
                    # double support leading to left swing, part 2 (mu < K_mu0 - pi)
                    else:
                        hip_R = -preswing + \
                            (preswing+prestance)*math.sin(0.5*math.pi *
                                                          (mu+2*math.pi-K_mu1)/(K_mu0-K_mu1+math.pi))
                        hip_L = -prestance + \
                            (prestance+preswing)*math.sin(0.5*math.pi *
                                                          (mu+2*math.pi-K_mu1)/(K_mu0-K_mu1+math.pi))
                        ank_R = presup+(prepro-presup)*(mu+2 *
                                                        math.pi-K_mu1)/(K_mu0-K_mu1+math.pi)
                        ank_L = prepro+(presup-prepro)*(mu+2 *
                                                        math.pi-K_mu1)/(K_mu0-K_mu1+math.pi)
        return hip_R, hip_L, ank_R, ank_L

    def leg_swing(self, A, phase, K_mu0, K_mu1):
        mu = phase
        K10 = 0.12  # lateral swing amplitude
        K11 = 0.1		# lateral swing amplitude offset
        K12 = 0.01  # turning lateral swing amplitude offset
        K13 = 0.750  # sagittal swing amplitude
        K14 = 0.4		# rotational swing amplitude
        K15 = 0.05  # rotational swing amplitude offset
        if mu < K_mu0:
            zeta_R = ((2*(mu+2*math.pi-K_mu1))/(2*math.pi-K_mu1+K_mu0))-1
        else:
            if mu < K_mu1:
                zeta_R = math.cos((math.pi*(mu-K_mu0)/(K_mu1-K_mu0)))
            else:
                zeta_R = ((2*(mu-K_mu1))/(2*math.pi-K_mu1+K_mu0))-1
        a1 = -zeta_R*A[0]*K10-max(abs(A[0])*K11, abs(A[2]*K12))
        a2 = zeta_R*A[1]*K13
        a3 = zeta_R*A[2]*K14-abs(A[2])*K15
        if mu < K_mu0-math.pi:
            zeta_L = ((2*(mu+3*math.pi-K_mu1))/(2*math.pi-K_mu1+K_mu0))-1
        else:
            if mu < K_mu1-math.pi:
                zeta_L = math.cos((math.pi*(mu+math.pi-K_mu0))/(K_mu1-K_mu0))
            else:
                zeta_L = ((2*(mu+math.pi-K_mu1))/(2*math.pi-K_mu1+K_mu0))-1
        b1 = -zeta_L*A[0]*K10+max(abs(A[0])*K11, abs(A[2]*K12))
        b2 = zeta_L*A[1]*K13
        b3 = zeta_L*A[2]*K14-abs(A[2])*K15
        RLegSwing = np.array([a1, a2, a3])
        LLegSwing = np.array([b1, b2, b3])
        return RLegSwing, LLegSwing

    def update_walk(self, vel, dt, phase):

        Sx = -vel[0]    	# step size (forward)
        Sy = vel[1]  # lateral step length
        Sz = vel[2]  # turning step magnitude
        # setting Sy,Sz to zero limits stepping to forward/backward only
        S = [Sx, Sy, Sz]
        K20 = 3.5
        K21 = 0.2
        K22 = 0.2
        K23 = 0.2
        Sy_min = 0.05
        Sy_max = 0.3
        kx = 0.3
        kphi = 0.2

        # transform desired step size S to step size in trunk frame 'St'
        Sx = math.cos(-0.5*S[2])*S[0]-math.sin(-0.5*S[2])*S[1]
        Sy = math.sin(-0.5*S[2])*S[0]+math.cos(-0.5*S[2])*S[1]
        St = [Sx, Sy, S[2]]

        # find swing amplitude target 'At'
        if -np.sign(phase)*St[1] > Sy_min:
            self.At1 = -np.sign(phase)*(abs(St[1])-Sy_min)/(Sy_max-Sy_min)
        else:
            self.At1 = self.At1

        At = np.array([self.At1, kx*St[0], kphi*St[2]])

        # if target swing amplitude is too large, set it to a lower value
        p = sum(abs(At)**K20)**(1./K20)
        if p > 1:
            At = At/p
            self.At1 = At[0]

        # move swing amplitude closer to target swing amplitude by bounded increments
        dA1 = min(max(At[0]-self.A[0], -K21*dt), K21*dt)
        dA2 = min(max(At[1]-self.A[1], -K22*dt), K22*dt)
        dA3 = min(max(At[2]-self.A[2], -K23*dt), K23*dt)
        dA = np.array([dA1, dA2, dA3])

        # update swing amplitude 
        A = self.A+dA
        self.A = A
        return A

    def get(self, A, phase):
        """ Obtain the joint angles """
        K_mu0 = 0.6  # swing start time
        K_mu1 = 3   # swing stop time
        K_mu2 = 2   # swing mid-time
        a1, a2 = self.eta_leg_lift(A, phase, K_mu0, K_mu1, K_mu2)
        b1, b2, c1, c2 = self.hip_ankle(phase, K_mu0, K_mu1)
        d1, d2 = self.leg_swing(A, phase, K_mu0, K_mu1)
        eta = [self.eta_R+a1, self.eta_L+a2]
        phi_Rleg = self.phi_Rleg+np.array([b1, 0, 0])+d1
        phi_Lleg = self.phi_Lleg+np.array([b2, 0, 0])+d2
        phi_Rfoot = self.phi_Rfoot+np.array([c1, 0])
        phi_Lfoot = self.phi_Lfoot+np.array([c2, 0])

        # eta = 1 denotes fully retracted leg (so eta should not be greater than 1)
        # eta = 0 denotes fully extended leg (so eta must not be negative, and the following lines ensure this):
        if eta[0] < 0:
            eta[0] = 0

        if eta[1] < 0:
            eta[1] = 0

        # angles computed from leg extension parameters
        zeta_R = -math.acos(1-eta[0])
        zeta_L = -math.acos(1-eta[1])

        # leg yaw values are taken directly from 'phi_leg'
        Rlegyaw = phi_Rleg[2]
        Llegyaw = phi_Lleg[2]

        # compute alternate pitch and leg angles
        Rlegpitch = math.cos(-Rlegyaw) * \
            phi_Rleg[1]-math.sin(-Rlegyaw)*phi_Rleg[0]
        Rlegroll = math.sin(-Rlegyaw) * \
            phi_Rleg[1]+math.cos(-Rlegyaw)*phi_Rleg[0]
        Llegpitch = math.cos(-Llegyaw) * \
            phi_Lleg[1]-math.sin(-Llegyaw)*phi_Lleg[0]
        Llegroll = math.sin(-Llegyaw) * \
            phi_Lleg[1]+math.cos(-Llegyaw)*phi_Lleg[0]

        # set leg joint angles
        angles = {}
        angles["l_ankle_frontal"] = -(phi_Lfoot[0]+Llegroll)
        angles["l_ankle_sagittal"] = -(phi_Lfoot[1]-Llegpitch-zeta_L)
        angles["l_knee"] = 2*zeta_L
        angles["l_hip_sagittal"] = (Llegpitch-zeta_L)
        angles["l_hip_frontal"] = -(Llegroll)
        angles["l_hip_swivel"] = Llegyaw
        angles["r_hip_swivel"] = Rlegyaw
        angles["r_hip_frontal"] = -(Rlegroll)
        angles["r_hip_sagittal"] = -(Rlegpitch-zeta_R)
        angles["r_knee"] = -2*zeta_R
        angles["r_ankle_sagittal"] = phi_Rfoot[1]-Rlegpitch-zeta_R
        angles["r_ankle_frontal"] = phi_Rfoot[0]-Rlegroll

        # set arm joint angles
        sag = -math.pi/8  # max sagittal shoulder joint displacement
        elb = math.pi/6  # max elbow joint displacement
        mu = phase-0.6  # offset arm motion phase by pi to alternate arm swinging appropriately

        angles["l_shoulder_frontal"] = -math.pi/9
        angles["r_shoulder_frontal"] = -math.pi/9
        if sum(abs(A)) > 0:  # if desired walk speed is greater than zero, then move arms
            angles["l_shoulder_sagittal"] = sag*math.sin(mu)
            angles["l_elbow"] = elb*math.cos(mu+math.pi)-elb
            angles["r_shoulder_sagittal"] = sag*math.sin(mu)
            angles["r_elbow"] = elb*math.cos(mu+math.pi)+elb
        else:  # otherwise, set arms to these angles, when no walking is desired
            angles["l_shoulder_sagittal"] = 0.3927
            angles["l_elbow"] = -0.5236
            angles["r_shoulder_sagittal"] = 0.3927
            angles["r_elbow"] = 0.5236
        
        f1 = angles["r_shoulder_sagittal"]
        f2 = angles["l_shoulder_sagittal"]
        f3 = angles["r_shoulder_frontal"]
        f4 = angles["l_shoulder_frontal"]
        f5 = angles["r_elbow"]
        f6 = angles["l_elbow"]
        f7 = angles["r_hip_swivel"]
        f8 = angles["l_hip_swivel"]
        f9 = angles["r_hip_frontal"]
        f10 = angles["l_hip_frontal"]
        f11 = angles["r_hip_sagittal"]
        f12 = angles["l_hip_sagittal"]
        f13 = angles["r_knee"]
        f14 = angles["l_knee"]
        f15 = angles["r_ankle_sagittal"]
        f16 = angles["l_ankle_sagittal"]
        f17 = angles["r_ankle_frontal"]
        f18 = angles["l_ankle_frontal"]
        
        f = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]) 
        
        return f

    def generate(self):
        """
        Populate joints, initial pose for walking with 18-DOF robot
        """
        # set leg joint angles
        angles = {}

        f = [-0.2, -1.1932, -1.7264, 0.4132, -0.15, 0, 0, 0.15, -0.4132, 1.7264,
             1.1932, 0.2, -0.3927, -0.3491, -0.5236, 0.3927, -0.3491, 0.5236]

        angles["l_ankle_frontal"] = f[0]
        angles["l_ankle_sagittal"] = f[1]
        angles["l_knee"] = f[2]
        angles["l_hip_sagittal"] = f[3]
        angles["l_hip_frontal"] = f[4]
        angles["l_hip_swivel"] = f[5]
        angles["r_hip_swivel"] = f[6]
        angles["r_hip_frontal"] = f[7]
        angles["r_hip_sagittal"] = f[8]
        angles["r_knee"] = f[9]
        angles["r_ankle_sagittal"] = f[10]
        angles["r_ankle_frontal"] = f[11]
        angles["l_shoulder_sagittal"] = f[12]
        angles["l_shoulder_frontal"] = f[13]
        angles["l_elbow"] = f[14]
        angles["r_shoulder_sagittal"] = f[15]
        angles["r_shoulder_frontal"] = f[16]
        angles["r_elbow"] = f[17]
        
        g = np.array([f[15],f[12],f[16],f[13],f[17],f[14],f[6],f[5],f[7],f[4],f[8],f[3],f[9],f[2],f[10],f[1],f[11],f[0]])
        
        return g


class Walker:
    """
    Class for making Tobe walk
    """

    def __init__(self, tobe):
        self.tobe=tobe
        self.running = False

        self.velocity = [0, 0, 0]
        self.walking = False
        self.func = WalkFunc()
        self.Tinit = 2
        self.dt = 0.02 # 
        self.T = self.Tinit
        self.phase = -math.pi
        self.A = self.func.update_walk(self.velocity, self.dt, self.phase)
        self.ready_pos = self.func.generate() #self.func.get(self.A, self.phase)
        self.angles = self.ready_pos

        self._th_walk = None

        self._sub_cmd_vel = rospy.Subscriber(tobe.ns+"cmd_vel", Vector3, self._cb_cmd_vel, queue_size=1)

    def _cb_cmd_vel(self, msg):
        """
        Catches cmd_vel and update walker speed
        """
        print("cmdvel", msg)
        vx = msg.x
        vy = msg.y
        vt = msg.z
        self.start()
        self.set_velocity(vx, vy, vt)

    def init_walk(self):
        """
        If not there yet, go to initial walk position
        """
        rospy.loginfo("Going to walk position")
        if self.get_dist_to_ready() > 0.02:
            self.set_angles_slow(self.ready_pos)
            rospy.loginfo("Done")

    def start(self):
        if not self.running:
            self.running = True
            self.init_walk()
            self._th_walk = Thread(target=self._do_walk)
            self._th_walk.start()
            self.walking = True

    def stop(self):
        if self.running:
            self.walking = False
            rospy.loginfo("Waiting for stopped")
            while not rospy.is_shutdown() and self._th_walk is not None:
                rospy.sleep(0.1)
            rospy.loginfo("Stopped")
            self.running = False

    def set_velocity(self, x, y, t):
        self.velocity = [x, y, t]

    def _do_walk(self):
        """
        Main walking loop, smoothly update velocity vectors and apply corresponding angles
        """
        # increment=1
        samplerate = 50 # 
        r = rospy.Rate(samplerate)
        dt = 1.0/samplerate
        rospy.loginfo("Started walking thread")
        func = self.func
        K25 = 3  # bound on step frequency omega

        # Global walk loop
        n = 50
        self.current_velocity = [0, 0, 0]
        while not rospy.is_shutdown() and (self.walking or self.is_walking()):
            if not self.walking:
                self.velocity = [0, 0, 0]
            if not self.is_walking():  # Do not move if nothing to do and already at 0
                self.update_velocity(self.velocity, n)
                r.sleep()
                continue
            # update time to foot touchdown T:
            self.T = self.T-dt  # decrement time until swing foot touchdown by 'dt'
            # if this value reaches zero (the foot should be touching surface)
            if self.T <= 0:
                self.T = self.Tinit

            # compute step frequency 'omega' based on 'prevphase' and 'T'
            if self.phase > 0:
                nu = math.pi-self.phase
            else:
                nu = -self.phase
            omega = nu/(self.T*math.pi)

            # omega should be bounded, since it can get large when T approaches zero
            if omega > K25:
                omega = K25

            # update phase ('phase' progresses from -pi to pi, then resets back at -pi
            self.phase = self.phase+omega*math.pi*dt

            # reset 'phase' to -pi, if necessary
            if self.phase >= math.pi:
                self.phase = -math.pi

            self.update_velocity(self.velocity, n)
            self.A = func.update_walk(self.velocity, dt, self.phase)
            
            self.angles = func.get(self.A, self.phase)
            self.set_angles(self.angles)
            
            r.sleep()
        rospy.loginfo("Finished walking thread")

        self._th_walk = None

    def is_walking(self):
        e = 0.02
        for v in self.current_velocity:
            if abs(v) > e:
                return True
        return False

    def rescale(self, angles, coef):
        z = {}
        for j, v in angles.items():
            offset = self.ready_pos[j]
            v -= offset
            v *= coef
            v += offset
            z[j] = v
        return z

    def update_velocity(self, target, n):
        a = 3/float(n)
        b = 1-a
        self.current_velocity = [
            a*t+b*v for (t, v) in zip(target, self.current_velocity)]

    def get_dist_to_ready(self):
        ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] # Dynamixel motor ID #'s
        poses = self.tobe.read_all_motor_positions()
        angles = self.tobe.convert_motor_positions_to_angles(ids, poses)
        return get_distance(self.ready_pos, angles)
        
    def set_angles_slow(self,stop_angles,delay=2):
        start_angles=self.angles
        start=time.time()
        stop=start+delay
        r=rospy.Rate(100)
        while not rospy.is_shutdown():
            t=time.time()
            if t>stop: break
            ratio=(t-start)/delay            
            angles=interpolate(stop_angles,start_angles,ratio)                        
            self.set_angles(angles)
            r.sleep()
            
    def set_angles(self, angles):
        ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] # Dynamixel motor ID #'s
        cmds = self.tobe.convert_angles_to_commands(ids, angles)            
        self.tobe.command_all_motors(cmds)


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #anglesa.keys()
    for j in joints:
        z[j] = anglesa[j]*coefa+anglesb[j]*(1-coefa)
    return z



def get_distance(anglesa, anglesb):
    d = 0
    joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #anglesa.keys()
    if len(joints) == 0:
        return 0
    for j in joints:
        d += abs(anglesb[j]-anglesa[j])
    d /= len(joints)
    return d



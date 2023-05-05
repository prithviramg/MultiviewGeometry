import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from pytransform3d.plot_utils import make_3d_axis
import matplotlib.cm as cm

IMAGE_SIZE = (640, 480)

# Define the Streamlit app
def app():
    # Set the page title
    st.set_page_config(page_title="Camera Parameters Visualisation", layout="wide", initial_sidebar_state="expanded")
    col1, col2 = st.columns(2)
    with col1:
        azimuth = st.slider("Azimuth", -180, 180, -35, 1)
    with col2:
        elevation = st.slider("Elevation", -180, 180, 20, 1)
    fig = plt.figure(figsize=(10, 5))
    ax1 = make_3d_axis(1, 121, unit="m")
    ax1.view_init(elevation, azimuth)
    ax2 = plt.subplot(122, aspect="equal")
    ax2.set_title("Camera image")
    ax2.set_xlim(0, IMAGE_SIZE[0])
    ax2.set_ylim(0, IMAGE_SIZE[1])

    ######################## WORLD CO-ORDINATES ###############################
    world_grid = pc.make_world_grid(n_lines=7, n_points_per_line= 19, xlim=(-1,1), ylim=(-1,1))
    world_grid = pt.transform(pt.transform_from(R=pr.active_matrix_from_intrinsic_euler_xyz(e=[np.pi/2,0,0]), p = [0,2,0]),
                            world_grid)
    colors = cm.rainbow(np.linspace(0, 1, len(world_grid)))
    ax1.scatter(
    world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=0.2, alpha=1, color=colors)
    ax1.scatter(world_grid[-1, 0], world_grid[-1, 1], world_grid[-1, 2], color="r")
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")
    ax1.set_zlabel("Z axis")
    ax1.set_xlim((-3,3))
    ax1.set_ylim((-3,3))
    ax1.set_zlim((-3,3))

    ######################## WORLD AXIS ###############################                  
    pt.plot_transform(ax1, A2B = pt.transform_from(R=pr.active_matrix_from_intrinsic_euler_xyz(e=[0,0,0]), p = [2.8,-2.8,-2.8]),
                  s = 1, name="world")

    ######################## CAMERA ###############################  
    with st.sidebar:
        st.subheader('CameraExtrinsic')
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            cam_x = st.slider(label = "Camera X Position", min_value = -3.0,max_value=  3.0, value= 0.0,step= 0.1, key="cam_x")
            cam_y = st.slider(label = "Camera Y Position", min_value = -3.0,max_value=  3.0, value= 0.0,step= 0.1, key="cam_y")
            cam_z = st.slider(label = "Camera Z Position", min_value = -3.0,max_value=  3.0, value= 0.0,step= 0.1, key="cam_z")
        with col2:
            cam_pitch = st.slider(label = "Camera Pitch", min_value = -180,max_value=  180, value= -90,step= 1, key="cam_pitch")
            cam_roll = st.slider(label = "Camera Roll", min_value = -180,max_value=  180, value= 0,step= 1, key="cam_roll")
            cam_yaw = st.slider(label = "Camera Yaw", min_value = -180,max_value=  180, value= 0,step= 1, key="cam_yaw")
    with st.sidebar:
        st.subheader('CameraIntrisic')
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            img_width = st.slider(label = "Image Width", min_value = 640, max_value=  1024, value= 640, key="img_width")
            img_height = st.slider(label = "Image Height", min_value = 480,max_value=  1024, value= 480, key="img_height")
            focal_length = st.slider(label = "focal length", min_value = 0.05,max_value=  3.0, value= 0.08,step= 0.05, key="focal_length")
        with col2:
            sensor_width = st.slider(label = "sensor width", min_value = 0.01,max_value=  0.1, value= 0.03,step= 0.01, key="sensor_width")
            sensor_height = st.slider(label = "sensor height", min_value = 0.01,max_value=  0.1, value= 0.02,step= 0.01, key="sensor_height")
            virtual_image_distance = st.slider(label = "virtual image distance", min_value = 0.5,max_value= 2.0, value= 0.5,step= 0.1, 
                                                key="virtual_image_distance")
    
    cam2world = pt.transform_from(R=pr.active_matrix_from_intrinsic_euler_xyz(e=[np.deg2rad(cam_pitch),
                                                                                 np.deg2rad(cam_yaw),
                                                                                 np.deg2rad(cam_roll)]), 
                                  p = [cam_x,cam_y,cam_z])
    sensor_size = np.array([st.session_state.sensor_width, st.session_state.sensor_height])
    image_size = (st.session_state.img_width, st.session_state.img_height)
    intrinsic_matrix = np.array([
        [st.session_state.focal_length,                             0, st.session_state.sensor_width / 2.0],
        [0,                             st.session_state.focal_length, st.session_state.sensor_height / 2.0],
        [0,                                                         0,                                    1]
    ])
    pt.plot_transform(ax1, A2B=cam2world, s=0.3, name="Camera")
    pc.plot_camera(
        ax1, cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size,
        virtual_image_distance=st.session_state.virtual_image_distance)
    
    ######################## IMAGE ###############################
    image_grid = pc.world2image(world_grid, cam2world, sensor_size, image_size,
                                st.session_state.focal_length, kappa=0)
    ax2.scatter(image_grid[:, 0], -(image_grid[:, 1] - image_size[1]), c = colors)
    ax2.scatter(image_grid[-1, 0], -(image_grid[-1, 1] - image_size[1]), color="r")
    ax2.set_xlim(0, st.session_state.img_width)
    ax2.set_ylim(0,st.session_state.img_height)
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Show the plot in Streamlit
    st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    app()

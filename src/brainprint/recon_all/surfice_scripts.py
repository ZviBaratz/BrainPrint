BILATERAL_MEDIAL_SURFACE_SCRIPT = """
import gl
gl.resetdefaults()

# Load bilateral brain meshes
gl.meshloadbilateral('BrainMesh_ICBM152.lh.mz3')

# Hide orientation cube
gl.orientcubevisible(0)

# Load data
gl.overlayload('{left_nii_path}')
gl.overlayload('{right_nii_path}')

# Specify colormap and common range
gl.overlaycolorname(1, "viridis")
gl.overlaycolorname(2, "viridis")
gl.overlayminmax(1, -{vmax}, {vmax})
gl.overlayminmax(2, -{vmax}, {vmax})

# Set up medial view and save
gl.azimuthelevation(180, 0)
gl.hemispheredistance(1.2)
gl.hemispherepry(-80)
gl.fullscreen(1)
gl.cameradistance(1.2)
gl.savebmp('{destination}')
# gl.quit()
"""

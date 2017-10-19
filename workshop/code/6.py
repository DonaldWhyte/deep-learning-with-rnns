# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 6:
# Using Saved Models to Generate Text
# ==============================================================================

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

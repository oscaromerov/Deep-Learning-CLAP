from transformers import ClapConfig, ClapModel

# Initializing a ClapConfig with laion-ai/base style configuration
configuration = ClapConfig()

# Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
model = ClapModel(configuration)
print(model)

# Accessing the model configuration
configuration = model.config
print(configuration)

# # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
# from transformers import ClapTextConfig, ClapAudioConfig
#
# # Initializing a ClapText and ClapAudioConfig configuration
# config_text = ClapTextConfig()
# config_audio = ClapAudioConfig()
#
# config = ClapConfig.from_text_audio_configs(config_text, config_audio)
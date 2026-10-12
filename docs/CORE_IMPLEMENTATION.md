CLIPs/
    --__init__.py
    # CLIP abstract class
    # property method 'to_be_converted': the attribute of the module to be converted to SNN. default 'visual'
    # property method 'conversion_plan': here only provide things related to ANN, such as the name of relu module. Do not pass SNN time, or neuron name here.
    --openai_rn50/
        # The model is downloaded by huggingface.
        # return a instance of CLIP abstract class with **loaded** weight.
        # weights are downloaded from huggingface at home.
        # Expose the OpenAI CLIP preprocessing transform separately; it is not
        # part of the model or the converted SNN artifact.
        # conversion_plan: for 'to_be_converted', **despite the attentionpool**, module names of all ReLUs, etc.

to_snn/
    --__init__.py:
    # SNNArtifact abstract class; with reset_state(), load() and save().
    # ConversionParams: a dataclass that restores the params required by SNN, such as time step, conversion method
    # expose the interface (a factory actually) to get SNNArtifact. input: clip='openai_rn50', conversion method, ConversionParams, dataloader.
        import CLIPs/{clip} to get loaded clip.
        import ./{conversion_method} to get the conversion API.
        only pass ann = get_attr(clip, clip.to_be_converted) to conversion API and get the SNN.
        # The returned artifact is the converted visual tower only. It accepts
        # one preprocessed image batch per call and returns the final visual
        # feature for that timestep. The caller owns reset_state() and the
        # timestep loop; the CLIP text tower remains ANN.
        # Dataloaders apply resize/crop/to-tensor/CLIP normalization before
        # yielding image batches. The model and artifact are tensor-in/tensor-out.
        # expose the interface (a factory) to load the saved SNNArtifact.
        # save/load use a self-contained PyTorch checkpoint containing the
        # architecture identifier, conversion parameters, metadata, and state.
    --conversion_param_templates/
        DCGS.yaml
    --DCGS/ 
        --__init__.py 
        # wrap all the logic into a function Convert_by_DCGS
        --files/
        # DCGSConversionParams(ConversionParams) class for required params
        # SNNArtifact_DCGS
        # ANN activation info collection: receive a CLIP visual tower and an
        # image-only dataloader yielding already CLIP-preprocessed tensors; save statistics and metadata to
        # to_snn/DCGS/{model}_{stat_method}_{UTC timestamp}/. The cache name
        # is generated automatically from the model name, percentile/variance
        # method, and timestamp; existing directories are rejected.
        # convert ANN to SNN using the saved statistics directory, the
        # CLIP.conversion_plan(), and DCGSConversionParams; return a visual
        # SNNArtifact. Thresholds are computed during calibration, not supplied
        # by the ANN adapter.

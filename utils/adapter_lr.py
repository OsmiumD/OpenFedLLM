def group_parameters_by_layer(model, global_lr, client_lr, global_name, client_name):
    client_params = []
    global_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:

            if global_name in name:
                global_params.append(param)
            elif client_name in name:
                client_params.append(param)

    params_group = [
        {'params': client_params, 'lr': client_lr},
        {'params': global_params, 'lr': global_lr},
    ]

    return params_group
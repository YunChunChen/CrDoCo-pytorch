def create_model(opt):

    print(opt.model)

    if opt.model == 'supervised':
        from .TaskModel import TNetModel
        s_model = TNetModel()
        t_model = TNetModel()
        s_model.initialize(opt)
        t_model.initialize(opt)
        print("source model [%s] was created." % (s_model.name()))
        print("target model [%s] was created." % (t_model.name()))
        return s_model, t_model

    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
        model.initialize(opt)
        print("model [%s] was created." % (model.name()))
        return model

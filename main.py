import fire
def discrete_multi():
    from viz.DiscreteViz import test
    test()

def single_agent():
    from viz.BadViz import main as main_single_agent
    main_single_agent()

if __name__ == '__main__':
    fire.Fire()

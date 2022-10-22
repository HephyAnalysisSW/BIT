import os
if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    data_directory         = "/groups/hephy/cms/robert.schoefbeck/NN/data/"
elif os.environ['USER'] in ['oskar.rothbacher']:
    plot_directory         = "/groups/hephy/cms/oskar.rothbacher/www/pytorch/"
    model_directory        = "/groups/hephy/cms/oskar.rothbacher/NN/models/"
    data_directory         = "/groups/hephy/cms/oskar.rothbacher/NN/data/"
elif os.environ['USER'] in ['lena.wild']:
    plot_directory = './plots/'
    #plot_directory         = "/groups/hephy/cms/lena.wild/www/pytorch/"
    model_directory        = "/groups/hephy/cms/lena.wild/NN/models/"
    data_directory         = "/groups/hephy/cms/lena.wild/NN/data/"
else:
    plot_directory = './plots'
    model_directory= "./trained_models"
    data_directory = "./cached_data"

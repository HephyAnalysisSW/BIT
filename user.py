import os

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory                      = "/groups/hephy/cms/robert.schoefbeck/www/BIT/"
    model_directory                     = "/groups/hephy/cms/robert.schoefbeck/BIT/models/"
elif os.environ['USER'] in ['rosmarie.schoefbeck']:
    plot_directory                      = "/groups/hephy/cms/rosmarie.schoefbeck/www/BIT/"
elif os.environ['USER'] in ['nikolaus.frohner']:
    plot_directory                      = "/groups/hephy/cms/nikolaus.frohner/www/etc/"
elif os.environ['USER'] in ['dennis.schwarz']:
    plot_directory                      = "/groups/hephy/cms/dennis.schwarz/www/BIT/"
elif os.environ['USER'] in ['suman.chatterjee']:
	plot_directory                      = "/groups/hephy/cms/suman.chatterjee/www/BIT/"
elif os.environ['USER'] in ['lukas.lechner']:
	plot_directory                      = "/groups/hephy/cms/lukas.lechner/www/BIT/"
elif os.environ['USER'] in ['stefan.rohshap']:
	plot_directory                      = "/groups/hephy/cms/stefan.rohshap/www/BIT/"
else:
    print "New user %s. Please add plot_directory in user.py" % os.environ['USER']
    plot_directory = "."

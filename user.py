import os

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory                      = "/mnt/hephy/cms/robert.schoefbeck/www/BIT/"
elif os.environ['USER'] in ['rosmarie.schoefbeck']:
    plot_directory                      = "/mnt/hephy/cms/rosmarie.schoefbeck/www/BIT/"
elif os.environ['USER'] in ['nikolaus.frohner']:
    plot_directory                      = "/mnt/hephy/cms/nikolaus.frohner/www/etc/"
else:
    print "New user %s. Please add plot_directory in user.py" % os.environ['USER']
    plot_directory = "."

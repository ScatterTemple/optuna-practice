# setup
## orca for fig.save_image
Windows
- access https://github.com/plotly/orca/releases
- Extract the windows-release.zip file.
- In the release folder, double-click on orca Setup X.Y.Z, this will create an orca icon on your Desktop.
- Right-click on the orca icon and select Properties from the context menu.
- From the Shortcut tab, copy the directory in the Start in field.
- Add this Start in directory to you system PATH (see below).
- Open a new Command Prompt and verify that the orca executable is available on your PATH.

> orca --help
Plotly's image-exporting utilities

  Usage: orca [--version] [--help] <command> [<args>]
  ...
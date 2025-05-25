(use-modules (guix packages))
(use-modules (guix gexp))
(use-modules (guix build-system glib-or-gtk))
(use-modules (guix build-system pyproject))
(use-modules (guix build-system gnu))
(use-modules (gnu packages gnome))
(use-modules (gnu packages maths))
(use-modules (gnu packages python-build))
(use-modules (gnu packages python-graphics))
(use-modules (gnu packages check))
(use-modules (gnu packages python-check))
(use-modules (gnu packages python-xyz))
(use-modules (gnu packages glib))
(use-modules (gnu packages gtk))
(use-modules (gnu packages mail))
(use-modules (gnu packages pkg-config))
(use-modules ((guix licenses) #:prefix license:))

(define %source-dir (getcwd))

(define-public crystal-viewer
  (package
    (name "crystal-viewer")
    (version "0.0.0")
    (source (local-file %source-dir
                        #:recursive? #t))
    (build-system pyproject-build-system)
    (arguments
     (list #:tests? #f))
    (native-inputs
     (list pkg-config python-setuptools python-wheel python-pytest))
    (propagated-inputs
     (list python-numpy
     python-pyopengl
     ; Crashes in numpy format handler: python-pyopengl-accelerate 
     python-pygobject
     gtk+ glib gobject-introspection
     python-pyglm))
    (synopsis "m3")
    (description "FIXME")
    (home-page "FIXME")
    (license license:mpl2.0)))
crystal-viewer

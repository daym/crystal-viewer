(load "guix.scm")

(use-modules (gnu packages gtk))
(use-modules (gnu packages glib))

(packages->manifest
  (cons* crystal-viewer
         (specifications->packages
          (list "python"
                ;"python-glfw" ; TODO: Is glfw needed ?
                ;"python-pygobject" ;"python-pyopengl" "python-pyopengl-accelerate"
                ;"gobject-introspection"
                "sed" ; someone, not us, uses this.
                ))))
;; python-glcontext
;; python-glue-core

;; TODO: python-vispy !!!!!!!!!!!!!!!!
;; TODO: python-pyglm or numpy

/* Bambu Studio 를 화면 없이 돌릴 때 미리보기 그림(썸네일)을 그리게 하는 다리.
   Bambu Studio 는 OSMesa(화면 없는 OpenGL)로 그림을 그리려 하는데, 안에 든 GLEW 는 GLX 용이라 함수 주소를 GLX 에 묻는다.
   그 물음을 OSMesa 로 돌려 준다.   gcc -shared -fPIC -o glshim.so glshim.c   (libOSMesa.so.8 과 같이 LD_PRELOAD) */
#include <string.h>
#include <stddef.h>
typedef void (*fn)(void);
extern fn OSMesaGetProcAddress(const char *name);
static int fake_display;
static void *glx_current_display(void) { return &fake_display; }
static void *glx_null(void) { return NULL; }
fn glXGetProcAddressARB(const unsigned char *name)
{
    const char *n = (const char *)name;
    if (!strcmp(n, "glXGetCurrentDisplay")) return (fn)glx_current_display;
    if (!strncmp(n, "glX", 3)) return (fn)glx_null;
    return OSMesaGetProcAddress(n);
}
fn glXGetProcAddress(const unsigned char *name) { return glXGetProcAddressARB(name); }
int glXQueryVersion(void *dpy, int *major, int *minor) { if (major) *major = 1; if (minor) *minor = 4; return 1; }
const char *glXGetClientString(void *dpy, int name) { return ""; }
const char *glXQueryExtensionsString(void *dpy, int screen) { return ""; }

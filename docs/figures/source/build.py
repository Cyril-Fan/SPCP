"""Build the confirmed SPCP figures from self-contained, editable SVG sources."""
from pathlib import Path
import argparse
import ctypes.util
import os
import shutil
import zipfile

SOURCE = Path(__file__).resolve().parent
ASSETS = SOURCE.parent


def open_library(ffi, variable, names):
    candidates = []
    if os.environ.get(variable):
        candidates.append(os.environ[variable])
    directories = os.environ.get('CAIROCFFI_DLL_DIRECTORIES', '').split(os.pathsep)
    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(found)
        candidates.append(name)
        for directory in directories:
            if directory:
                candidates.append(str(Path(directory) / name))
    for candidate in dict.fromkeys(candidates):
        try:
            return ffi.dlopen(candidate)
        except OSError:
            pass
    raise RuntimeError(f'Cannot load {variable}. Set it to the installed library path.')


def export_pdfs(package):
    import cairosvg
    import cairocffi as cairo
    from cffi import FFI

    # CairoSVG does not load SVG web fonts. Use the supplied font directly
    # through FreeType, without installing or changing any system font.
    ffi = FFI()
    ffi.cdef('typedef void* FT_Library; typedef void* FT_Face; '
             'int FT_Init_FreeType(FT_Library*); '
             'int FT_New_Face(FT_Library,const char*,long,FT_Face*); '
             'void *cairo_ft_font_face_create_for_ft_face(FT_Face,int);')
    ft = open_library(ffi, 'SPCP_FREETYPE_LIBRARY',
                      ['freetype', 'freetype.dll', 'libfreetype-6.dll', 'libfreetype.so.6', 'libfreetype.dylib'])
    clib = open_library(ffi, 'SPCP_CAIRO_LIBRARY',
                        ['cairo', 'cairo-2.dll', 'libcairo-2.dll', 'libcairo.so.2', 'libcairo.2.dylib'])
    library = ffi.new('FT_Library*')
    face = ffi.new('FT_Face*')
    assert ft.FT_Init_FreeType(library) == 0
    font_path = package / 'editable/fonts/SPCPNewTXMathI.otf'
    assert ft.FT_New_Face(library[0], str(font_path).encode(), 0, face) == 0
    pointer = clib.cairo_ft_font_face_create_for_ft_face(face[0], 0)
    fontface = cairo.FontFace._from_pointer(
        cairo.ffi.cast('cairo_font_face_t*', int(ffi.cast('uintptr_t', pointer))), incref=False)
    original = cairo.Context.select_font_face

    def select_font(context, family, *args):
        if family == 'SPCP NewTX Math I':
            context.set_font_face(fontface)
        else:
            original(context, family, *args)

    cairo.Context.select_font_face = select_font
    try:
        for kind in ['core', 'overall']:
            cairosvg.svg2pdf(
                bytestring=(package / f'editable/SPCP_{kind}.svg').read_bytes(),
                write_to=str(package / f'figures/SPCP_{kind}.pdf'))
    finally:
        cairo.Context.select_font_face = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='A new, empty build directory.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory must be empty; existing deliverables are not overwritten.')
    package = output / 'SPCP_figures'
    (package / 'figures').mkdir(parents=True, exist_ok=True)
    (package / 'editable/fonts').mkdir(parents=True)
    for name in ['SPCP_overall.svg', 'SPCP_core.svg']:
        shutil.copyfile(ASSETS / name, package / 'editable' / name)
    shutil.copyfile(ASSETS / 'fonts/SPCPNewTXMathI.otf',
                    package / 'editable/fonts/SPCPNewTXMathI.otf')
    shutil.copyfile(SOURCE / 'PACKAGE_README.md', package / 'README.md')
    export_pdfs(package)
    names = ['README.md', 'figures/SPCP_overall.pdf', 'figures/SPCP_core.pdf',
             'editable/SPCP_overall.svg', 'editable/SPCP_core.svg',
             'editable/fonts/SPCPNewTXMathI.otf']
    actual = sorted(p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file())
    assert actual == sorted(names)
    archive = output / 'SPCP_structure_figures.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            zipped.write(package / name, 'SPCP_figures/' + name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name in names:
            assert zipped.read('SPCP_figures/' + name) == (package / name).read_bytes()
    print(archive)


if __name__ == '__main__':
    main()

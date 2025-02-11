# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include NLTK data
        ('C:\\Users\\Rayyan Khan\\AppData\\Roaming\\nltk_data', 'nltk_data'),
        # Include other data files if necessary
    ],
    hiddenimports=[
        'pandas',
        'numpy',
        'matplotlib',
        'tensorflow',
        'scikit-learn',
        'nltk',
        'newsapi',
        'torch',
        'transformers',
        'reportlab',
        'dateutil',
        'sklearn',
        'yaml',
        'requests',
        'pkg_resources.py2_warn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StockBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True to show console for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StockBot',
)

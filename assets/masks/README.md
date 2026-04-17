# Mask Assets

Place mask asset directories here. Each directory must contain a `mask.json` descriptor.

## Directory layout

```
assets/masks/
└── your_mask_name/
    ├── mask.json       # Required — descriptor file
    ├── texture.png     # For type: overlay_2d
    ├── avatar.glb      # For type: blendshape
    └── ...
```

## mask.json structure

### overlay_2d
```json
{
  "type": "overlay_2d",
  "texture": "texture.png",
  "anchors": {
    "left_eye":  [x, y],
    "right_eye": [x, y],
    "nose_tip":  [x, y]
  }
}
```

### blendshape
```json
{
  "type": "blendshape",
  "model": "avatar.glb",
  "expression_map": {
    "mouth_open": "jawOpen",
    "brow_raise": "browInnerUp"
  }
}
```

### filter
```json
{
  "type": "filter",
  "effects": ["cartoon"],
  "colour_grade": { "hue_shift": 10, "sat_scale": 1.3, "val_scale": 1.0 }
}
```

## Available effects (filter type)
| Effect | Description |
|---|---|
| `cartoon` | Bilateral smooth + adaptive threshold edges |
| `greyscale` | Desaturate the face region |
| `colour_grade` | HSV hue/saturation/value shift |
| `edge_glow` | Canny edge glow composited over original |

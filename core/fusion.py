"""core/fusion.py — cross-camera fusion into global player IDs (ACCURACY GATE).

NOT YET BUILT. Implemented in Step 6.

In the overlap zone, match by shared court position + jersey color to collapse
two camera-local tracks into one global ID. Shared global player registry is
protected by a threading.Lock.
"""

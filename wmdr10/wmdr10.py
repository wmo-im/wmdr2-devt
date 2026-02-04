from __future__ import annotations

import json
import numbers
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, Sequence

import xmltodict
import yaml

# from acdd.acdd import ACDD
# from utils.utils import load_mapping_csv

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore



class WMDR10:
    """
    WMDR10 provides a parser and serializer for WMO WMDR 1.0 metadata records.

    Supports input as a file path, raw XML string, or pre-parsed dictionary.
    """
    data: Any

    def __init__(self, source: str | Path | dict, *, source_type: str = "file", simplify: bool = True):
        """
        Initialize a WMDR10 instance from a file path, XML string, or dictionary.

        Args:
            source (str | Path | dict): The input data. Its interpretation depends on `source_type`:
                - 'file': path to XML file
                - 'xml': raw XML string
                - 'dict': already-parsed metadata dictionary
            source_type (str, optional): One of {'file', 'xml', 'dict'}.
                Determines how `source` is interpreted. Defaults to 'file'.
            simplify (bool, optional): Whether to simplify the parsed metadata. Defaults to True.

        Raises:
            ValueError: If `source_type` is invalid or parsing fails.
        """
        if source_type == "file":
            if not isinstance(source, (str, Path)):
                raise TypeError("When source_type='file', source must be a str or Path.")
            source_path = Path(source)
            with source_path.open("r", encoding="utf-8") as f:
                self.data = xmltodict.parse(f.read(), process_namespaces=False)

        elif source_type == "xml":
            # Accept either a raw XML string or a Path to an XML file; reject dicts.
            if isinstance(source, Path):
                source = source.read_text(encoding="utf-8")
            if not isinstance(source, str):
                raise TypeError("When source_type='xml', source must be an XML string or a Path.")
            self.data = xmltodict.parse(source, process_namespaces=False)

        elif source_type == "dict":
            if not isinstance(source, dict):
                raise ValueError("When using source_type='dict', source must be a dict.")
            self.data = source

        else:
            raise ValueError(f"Invalid source_type: {source_type!r}. Must be 'file', 'xml', or 'dict'.")

        if simplify:
            self._simplify()

    @staticmethod
    def _normalize_key(name: str) -> str:
        if not isinstance(name, str):
            return str(name)
        s = name.strip()
        # remove namespace (prefix:local → local)
        if ":" in s:
            s = s.split(":", 1)[1]
        # drop leading attribute/text sigils
        if s.startswith("@") or s.startswith("#"):
            s = s[1:]
        return s.lower()

    @staticmethod
    def _select_latest_description(val):
        from datetime import datetime
        def _parse_iso(ts):
            if not ts or not isinstance(ts, str):
                return None
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                return None

        def _desc(x):
            if isinstance(x, dict):
                return x.get('description') or x.get('Description') or x.get('#text')
            return x if isinstance(x, str) else None

        def _begin(x):
            if isinstance(x, dict):
                vp = x.get('validPeriod')
                if isinstance(vp, dict):
                    return _parse_iso(vp.get('beginPosition') or vp.get('begin'))
                # sometimes beginPosition lives on the same dict
                return _parse_iso(x.get('beginPosition'))
            return None

        if isinstance(val, list):
            best = None
            best_dt = None
            for el in val:
                d = _desc(el)
                if d is None:
                    continue
                dt = _begin(el)
                if best_dt is None or (dt is not None and dt > best_dt):
                    best_dt, best = dt, d
            if best is not None:
                return best
            # fallback: first stringy description in the list
            for el in val:
                d = _desc(el)
                if d:
                    return d
            return None

        return _desc(val)


    def _simplify(self) -> None:
        """
        1) Strip XML namespaces.
        2) Atomic simplifications (single-pass, recursive):
        - @xsi:nil → None
        - @xlink:href → value
        - ('@codeSpace', '#text') → '#text'
        - Drop '@codeListValue' (keep '@codeList' if present)
        - CharacterString → value
        - pos → value
        - linkage.URL → {'url': URL}
        - geoLocation: {'Point': '...'} or {'Point': {'pos': '...'}} or {'pos': '...'} → '...'
        - Hoist linkage.url / onlineResource.linkage.url → parent['url']
        - validPeriod: MERGE inner fields (e.g., beginPosition/endPosition) into parent
            (do NOT drop None values)
        3) Explicitly unwrap facility.ObservingFacility → merge into facility (even with siblings).
        4) Promote facility.observation → top-level 'observations' (list). Remove the source keys.
        5) ISO singletons unwrap (CI_*, MD_*, OM_*, DQ_*, EX_*, GM_*, LI_*, PT_*, RS_*, SV_*) anywhere.
        6) Generic same-name collapse (foo:{Foo:{...}} and list variants), run to fixed point.
        This also collapses responsibleParty → responsibleParty chains.
        7) Build flat observations (preserving info):
        - unwrap {'observingcapability': {...}} list items
        - expand inner 'observation' (dict or list); unwrap {'om_observation': {...}} inside
        - per inner observation:
            • drop featureOfInterest
            • drop phenomenonTime/resultTime if None or {}
            • unwrap procedure/process; collect deployment(s) case-insensitively anywhere under it;
            normalize deployment:
                - deployedEquipment → Equipment
                - dataGeneration → list
            • NEW: if result → ResultSet → distributionInfo → distributor exists,
                    hoist to top-level "distributor" and drop "result"
        8) Normalize “uniform list” wrappers across the tree:
        - programAffiliation: [{ '@xlink:href': url }, ...] → [url, ...]
        - electronicMailAddress: [{ 'CharacterString': v }, ...] → [v, ...]
        9) Rename keys (case-insensitive):
        - schedule → coverage  (safe merge)
        10) Final scrub: drop stray 'featureOfInterest' / 'om_observation'. Do NOT blanket-drop None.
        11) Rename headerInformation → header.
        """

        NAMESPACES_TO_STRIP = {
            'gml', 'xlink', 'wmdr', 'gco', 'gmd', 'gmlexr', 'om', 'metce', 'xsi', 'sams', 'sam'
        }
        ISO_PREFIXES = ("CI_", "MD_", "OM_", "DQ_", "EX_", "GM_", "LI_", "PT_", "RS_", "SV_")

        # ---------- small helpers ----------

        def keynorm(s):
            return ''.join(ch for ch in str(s).lower() if ch.isalnum())

        def same_name(a, b):
            return keynorm(a) == keynorm(b)

        def normalize_to_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
            return [val]

        def href_list_to_strings(val):
            if isinstance(val, list) and all(isinstance(x, dict) and set(x.keys()) == {'@xlink:href'} for x in val):
                return [x['@xlink:href'] for x in val]
            return val

        def charstring_list_to_strings(val):
            """Convert uniform [{'CharacterString': v}, ...] → [v, ...] (case-insensitive key)."""
            if not isinstance(val, list) or not val:
                return val
            out = []
            for x in val:
                if isinstance(x, dict) and len(x) == 1:
                    (k, v), = x.items()
                    if k.lower() == "characterstring":
                        out.append(v)
                    else:
                        return val
                else:
                    return val
            return out

        def is_iso_singleton(dct):
            if isinstance(dct, dict) and len(dct) == 1:
                (k, _), = dct.items()
                return any(k.startswith(p) for p in ISO_PREFIXES)
            return False

        def merge_into(dst, src):
            for k, v in src.items():
                dst[k] = v

        def drop_keys_ci(obj, keys_lower):
            if isinstance(obj, dict):
                for kk in list(obj.keys()):
                    if kk.lower() in keys_lower:
                        obj.pop(kk, None)
                    else:
                        drop_keys_ci(obj[kk], keys_lower)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    drop_keys_ci(obj[i], keys_lower)
            return obj

        def pop_ci(d: dict, name: str):
            """Pop a key case-insensitively; return value or None."""
            if not isinstance(d, dict):
                return None
            target = keynorm(name)
            for k in list(d.keys()):
                if keynorm(k) == target:
                    return d.pop(k)
            return None

        def get_ci(d: dict, name: str):
            """Get a value case-insensitively (without popping)."""
            if not isinstance(d, dict):
                return None
            target = keynorm(name)
            for k, v in d.items():
                if keynorm(k) == target:
                    return v
            return None

        def unwrap_single_key(obj):
            """Unwrap dicts like {'Foo': {...}} repeatedly → {...}."""
            while isinstance(obj, dict) and len(obj) == 1:
                (_, obj), = obj.items()
            return obj

        def collect_by_key_ci(obj, target_norm: str):
            """Collect values for keys whose normalized name == target_norm, at any depth."""
            out = []
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if keynorm(k) == target_norm:
                            out.append(v)
                        walk(v)
                elif isinstance(o, list):
                    for el in o:
                        walk(el)
            walk(obj)
            return out

        def rename_key_ci(obj, from_name: str, to_name: str):
            """Rename keys recursively, case-insensitive; safe-merge dict→dict."""
            if isinstance(obj, dict):
                # First recurse children
                for k in list(obj.keys()):
                    obj[k] = rename_key_ci(obj[k], from_name, to_name)
                # Then rename here
                for k in list(obj.keys()):
                    if keynorm(k) == keynorm(from_name):
                        val = obj.pop(k)
                        # Merge if to_name exists and both are dicts
                        if to_name in obj and isinstance(obj[to_name], dict) and isinstance(val, dict):
                            merge_into(obj[to_name], val)
                        elif to_name in obj and isinstance(obj[to_name], list) and isinstance(val, list):
                            obj[to_name] = obj[to_name] + val
                        else:
                            obj[to_name] = val
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = rename_key_ci(obj[i], from_name, to_name)
            return obj

        # ---------- domain helpers ----------

        def normalize_deployment(dep):
            """
            Normalize a deployment object:
            - unwrap {'Deployment': {...}} (and other single-key wrappers)
            - deployedEquipment → Equipment (unwrap inner wrapper if needed)
            - dataGeneration → list (case-insensitive)
            - NEW: drop identifier if it is exactly {'@codeSpace': ...}
            """
            dep = unwrap_single_key(dep)
            if not isinstance(dep, dict):
                return {'_unexpected_deployment': dep}

            # deployedEquipment → Equipment
            for k in list(dep.keys()):
                if keynorm(k) == 'deployedequipment':
                    de = unwrap_single_key(dep.pop(k))
                    if isinstance(de, dict) and 'Equipment' in de:
                        dep['Equipment'] = de['Equipment']
                    elif isinstance(de, dict):
                        for kk, vv in de.items():
                            dep[kk] = vv
                    else:
                        dep['Equipment'] = de
                    break

            # dataGeneration → list
            for k in list(dep.keys()):
                if keynorm(k) == 'datageneration':
                    dg = dep.pop(k)
                    dep['dataGeneration'] = [unwrap_single_key(x) if isinstance(x, dict) else x
                                            for x in (dg if isinstance(dg, list) else [dg])]
                    break

            # drop identifier == {'@codeSpace': ...}
            for k in list(dep.keys()):
                if keynorm(k) == 'identifier':
                    idv = dep[k]
                    if isinstance(idv, dict) and any(kk.lower().strip() in ('@codespace',) for kk in idv.keys()) and len(idv) == 1:
                        dep.pop(k, None)
                    break

            return dep

        def flatten_observation_dict(inner):
            """
            Flatten a single inner observation:
            - drop featureOfInterest
            - drop phenomenonTime/resultTime if None or {}
            - unwrap procedure/process; collect deployment(s) anywhere under it; normalize deployment objects
            - NEW: hoist result→ResultSet→distributionInfo→distributor → top-level 'distributor'; drop 'result'
            """
            if not isinstance(inner, dict):
                return {'_unexpected_observation': inner}

            inner.pop('featureOfInterest', None)

            # Drop null/empty times (keep real TimeInstant/TimePeriod objects)
            for tkey in ('phenomenonTime', 'resultTime'):
                if tkey in inner:
                    tv = inner[tkey]
                    if tv is None or (isinstance(tv, dict) and not tv):
                        inner.pop(tkey, None)

            # Extract procedure / process (case-insensitive) and remove it from inner
            proc = pop_ci(inner, 'procedure')
            if proc is None:
                proc = pop_ci(inner, 'process')

            deployments_all = []
            if proc is not None:
                for proc_item in (proc if isinstance(proc, list) else [proc]):
                    hits = collect_by_key_ci(proc_item, 'deployment')
                    for hit in hits:
                        for dpl in (hit if isinstance(hit, list) else [hit]):
                            deployments_all.append(unwrap_single_key(dpl))
                    if not hits:
                        cand = unwrap_single_key(proc_item)
                        if isinstance(cand, dict) and any(keynorm(k) in ('deployedequipment', 'datageneration') for k in cand.keys()):
                            deployments_all.append(cand)

            if deployments_all:
                inner['deployments'] = [normalize_deployment(d) for d in deployments_all]

            # ---- NEW: unwrap result → ResultSet → distributionInfo → distributor ----
            if 'result' in inner and isinstance(inner['result'], (dict, list)):
                # handle both dict and single-element list
                rnode = inner['result']
                if isinstance(rnode, list) and len(rnode) == 1:
                    rnode = rnode[0]
                rnode = unwrap_single_key(rnode) if isinstance(rnode, dict) else rnode
                if isinstance(rnode, dict):
                    # After unwrap, either we are at {'distributionInfo': {...}} or deeper
                    dset = rnode
                    # tolerate 'ResultSet' wrapper if still present
                    if get_ci(dset, 'ResultSet') is not None:
                        dset = get_ci(dset, 'ResultSet')
                    dset = unwrap_single_key(dset) if isinstance(dset, dict) else dset
                    if isinstance(dset, dict):
                        di = get_ci(dset, 'distributionInfo')
                        if isinstance(di, dict):
                            dist = get_ci(di, 'distributor')
                            if dist is not None:
                                inner['distributor'] = unwrap_single_key(dist)
                                inner.pop('result', None)  # remove only when successfully extracted

            return inner

        # ---------- core passes ----------

        def strip_ns(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if k.startswith('@xmlns') or k == 'xmlns':
                        continue
                    key = k
                    if ':' in key:
                        prefix, base = key.split(':', 1)
                        if prefix in NAMESPACES_TO_STRIP:
                            key = base
                    if '}' in key:
                        key = key.split('}', 1)[-1]
                    out[key] = strip_ns(v)
                return out
            if isinstance(obj, list):
                return [strip_ns(x) for x in obj]
            return obj

        def _extract_geolocation_value(v):
            """Return 'lat lon alt' from geoLocation dicts."""
            if isinstance(v, dict):
                if 'Point' in v:
                    p = v['Point']
                    if isinstance(p, dict) and 'pos' in p:
                        return p['pos']
                    return p
                if 'pos' in v:
                    return v['pos']
            return v

        def simplify_atomic_at(parent, key, val):
            """
            Atomic, local normalizations. May MERGE/HOIST into parent.
            - @xsi:nil → None
            - @xlink:href → scalar
            - ('@codeSpace', '#text') → '#text'
            - drop '@codeListValue' (keep '@codeList')
            - CharacterString → scalar
            - pos → scalar
            - geoLocation: {'Point': '...'} or {'Point': {'pos':'...'}} or {'pos':'...'} → scalar
            - **linkage.url/URL → parent['url']; drop 'linkage' or 'onlineResource'**
            - validPeriod: merge TimePeriod fields into parent (keep None values)
            """
            v = val
            if isinstance(v, dict):
                # nil → None
                if v.get('@xsi:nil') == 'true':
                    return None

                # inline xlink href
                if '@xlink:href' in v:
                    return v['@xlink:href']

                # codeSpace/#text → value
                if '@codeSpace' in v and '#text' in v:
                    return v['#text']

                # drop @codeListValue; keep @codeList if present
                if '@codeListValue' in v:
                    v = {kk: vv for kk, vv in v.items() if kk != '@codeListValue'}
                    if set(v.keys()) == {'@codeList'}:
                        return v['@codeList']

                # pos → value
                if list(v.keys()) == ['pos']:
                    return v['pos']

                # CharacterString → value
                if list(v.keys()) == ['CharacterString']:
                    return v['CharacterString']

                # geoLocation: {'Point': '...'} / {'Point': {'pos':'...'}} / {'pos': '...'} → '...'
                if keynorm(key) == 'geolocation':
                    if 'Point' in v:
                        p = v['Point']
                        if isinstance(p, dict) and 'pos' in p:
                            return p['pos']
                        return p
                    if 'pos' in v:
                        return v['pos']

                # --- Unified linkage handling ---
                # Hoist onlineResource.linkage.url → parent['url']
                if keynorm(key) in ('onlineresource', 'onlineresources'):
                    link = v.get('linkage')
                    if isinstance(link, dict):
                        if 'url' in link:
                            parent['url'] = link['url']
                            return '__DROPPED__'
                        if 'URL' in link:
                            parent['url'] = link['URL']
                            return '__DROPPED__'

                # Hoist linkage.url/URL → parent['url']
                if keynorm(key) == 'linkage':
                    if 'url' in v:
                        parent['url'] = v['url']
                        return '__DROPPED__'
                    if 'URL' in v:
                        parent['url'] = v['URL']
                        return '__DROPPED__'

                # validPeriod: MERGE fields into parent; KEEP None values
                if key.lower() == 'validperiod':
                    inner = v.get('TimePeriod', v)
                    if isinstance(inner, dict):
                        parent.pop(key, None)
                        for kk, vv in inner.items():
                            parent[kk] = vv
                        return '__DROPPED__'

            return v


        def atomic_pass(d):
            """Apply atomic simplifications recursively."""
            if not isinstance(d, dict):
                return d
            # focus WIGOSMetadataRecord if present
            if "WIGOSMetadataRecord" in d:
                d = d["WIGOSMetadataRecord"]

            # prune noisy attributes
            d.pop('@xsi:schemaLocation', None)
            # drop boundedBy and 'type':'simple' / '@xlink:type':'simple'
            d = {
                k: v for k, v in d.items()
                if not ((k == 'type' or k == '@xlink:type') and v == 'simple')
                and k.lower() != 'boundedby'
            }

            # atomic at this level
            for k in list(d):
                val = simplify_atomic_at(d, k, d[k])
                if val != '__DROPPED__':
                    d[k] = val
                else:
                    d.pop(k, None)

            # recurse
            for k in list(d):
                v = d[k]
                if isinstance(v, dict):
                    d[k] = atomic_pass(v)
                elif isinstance(v, list):
                    d[k] = [atomic_pass(x) if isinstance(x, dict) else x for x in v]

            return d

        def unwrap_observingfacility(root_obj):
            """Step 3: unwrap facility.ObservingFacility → merge into facility (even with siblings present)."""
            fac = root_obj.get('facility')
            if isinstance(fac, dict):
                for ik in list(fac.keys()):
                    if keynorm(ik) == 'observingfacility' and isinstance(fac[ik], dict):
                        child = fac.pop(ik)
                        merge_into(fac, child)

        def iso_unwrap_anywhere(obj):
            """Unwrap ISO singletons in dict values and list elements."""
            if isinstance(obj, dict):
                if is_iso_singleton(obj):
                    (_, inner), = obj.items()
                    return iso_unwrap_anywhere(inner)
                for k in list(obj.keys()):
                    v = obj[k]
                    if is_iso_singleton(v):
                        (_, inner), = v.items()
                        obj[k] = iso_unwrap_anywhere(inner)
                    else:
                        obj[k] = iso_unwrap_anywhere(v)
                return obj
            if isinstance(obj, list):
                new = []
                for el in obj:
                    if is_iso_singleton(el):
                        (_, inner), = el.items()
                        new.append(iso_unwrap_anywhere(inner))
                    else:
                        new.append(iso_unwrap_anywhere(el))
                return new
            return obj

        # Generic same-name collapse (dicts + list elements), run to fixed point
        def _collapse_same_name_wrappers(obj, parent_key=None):
            changed = False

            if isinstance(obj, dict):
                # Recurse first, passing child's key as parent_key
                for k in list(obj.keys()):
                    if _collapse_same_name_wrappers(obj[k], k):
                        changed = True

                # Then collapse same-name children inside each dict value (handles chains)
                for k in list(obj.keys()):
                    v = obj[k]
                    if isinstance(v, dict):
                        while True:
                            found = None
                            for ik, iv in list(v.items()):
                                if isinstance(iv, dict) and same_name(ik, k):
                                    found = (ik, iv)
                                    break
                            if not found:
                                break
                            ik, inner = found
                            v.pop(ik, None)
                            for kk, vv in inner.items():
                                v[kk] = vv
                            changed = True
                return changed

            if isinstance(obj, list):
                for i, el in enumerate(list(obj)):
                    progressed = False
                    while isinstance(el, dict) and len(el) == 1:
                        (ik, iv), = el.items()
                        if parent_key is not None and same_name(ik, parent_key):
                            el = iv
                            progressed = True
                            changed = True
                        else:
                            break
                    if progressed:
                        obj[i] = el
                    if _collapse_same_name_wrappers(obj[i], parent_key):
                        changed = True
                return changed

            return changed

        # ---------- pipeline (ordered) ----------

        # 1) strip namespaces
        self.data = strip_ns(self.data)

        # 2) atomic simplifications (recursive)
        self.data = atomic_pass(self.data)

        # 3) explicit unwrap of facility.ObservingFacility
        unwrap_observingfacility(self.data)

        # 4) promote facility.observation → observations (robust)
        fac = self.data.get('facility')
        if isinstance(fac, dict):
            obs_blocks = []
            if 'observation' in fac:
                obs_blocks.extend(normalize_to_list(fac['observation']))
            inner_of = fac.get('ObservingFacility')
            if isinstance(inner_of, dict) and 'observation' in inner_of:
                obs_blocks.extend(normalize_to_list(inner_of['observation']))
                inner_of.pop('observation', None)
            if obs_blocks:
                existing = self.data.get('observations')
                self.data['observations'] = (normalize_to_list(existing) + obs_blocks) if existing else obs_blocks
                fac.pop('observation', None)

        # 5) ISO unwrap anywhere (dicts & list elements)
        self.data = iso_unwrap_anywhere(self.data)

        # 6) generic same-name collapse (fixed-point)
        while _collapse_same_name_wrappers(self.data):
            pass

        # 7) Build flat observations
        root = self.data
        if isinstance(root, dict) and 'observations' in root and isinstance(root['observations'], list):
            built = []
            for item in root['observations']:
                # unwrap {"observingcapability": {...}} items
                if isinstance(item, dict) and len(item) == 1:
                    (only_k, only_v), = item.items()
                    if only_k.lower() == 'observingcapability':
                        item = only_v
                if not isinstance(item, dict):
                    built.append(item)
                    continue

                # normalize base-level programAffiliation (uniform href-lists → strings)
                if 'programAffiliation' in item:
                    item['programAffiliation'] = href_list_to_strings(item['programAffiliation'])

                inner_obs = item.get('observation')
                if inner_obs is None:
                    built.append(item)
                    continue

                for inner in normalize_to_list(inner_obs):
                    # unwrap {'om_observation': {...}} if present
                    if isinstance(inner, dict) and len(inner) == 1:
                        (ik, iv), = inner.items()
                        if ik.lower() == 'om_observation':
                            inner = iv
                    flat = flatten_observation_dict(inner) if isinstance(inner, dict) else {'_unexpected_observation': inner}
                    out = {k: v for k, v in item.items() if k != 'observation'}
                    out.update(flat)
                    built.append(out)

            root['observations'] = built

            # normalize observation-level deployments → 'deployments' (plural) and list-ify
            for obs in root['observations']:
                if not isinstance(obs, dict):
                    continue
                legacy = obs.pop('deployment', None)
                if legacy is not None:
                    legacy_list = legacy if isinstance(legacy, list) else [legacy]
                    if 'deployments' in obs:
                        current = obs['deployments']
                        current_list = current if isinstance(current, list) else [current]
                        obs['deployments'] = current_list + legacy_list
                    else:
                        obs['deployments'] = legacy_list
                elif 'deployments' in obs and not isinstance(obs['deployments'], list):
                    obs['deployments'] = [obs['deployments']]

        # Run same-name collapse once more post-expansion (cleans responsibleParty chains, etc.)
        while _collapse_same_name_wrappers(self.data):
            pass

        # 8) normalize uniform list wrappers everywhere
        def normalize_lists_inplace(obj):
            if isinstance(obj, dict):
                for k in list(obj.keys()):
                    v = obj[k]
                    if isinstance(v, list):
                        obj[k] = href_list_to_strings(v)
                        obj[k] = charstring_list_to_strings(obj[k])
                    normalize_lists_inplace(obj[k])
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = normalize_lists_inplace(obj[i])
            return obj
        normalize_lists_inplace(self.data)

        # 8b) Harmonize data shapes across records (stable downstream mapping)
        #
        # The XML source can legally contain repeated elements; xmltodict then produces either a scalar
        # (single occurrence) or a list (multiple occurrences). For a few elements we have observed both
        # shapes across your sample records and want a single, predictable representation:
        #   - programAffiliation: always a list of href strings
        #   - applicationArea: always a list of href strings
        #   - transferOptions.onLine: always a list of online resource objects
        #
        # In addition, some code-list elements are encoded in WMDR as empty xlink "stubs" like
        #   { "@xlink:type": "simple" }
        # which turn into {} after we drop the type. Those empty dict placeholders are removed here.
        ALWAYS_LIST_KEYS_NORM = {keynorm('programAffiliation'), keynorm('applicationArea'), keynorm('onLine')}

        def harmonize_shapes(obj):
            # recurse + normalize
            if isinstance(obj, dict):
                for k in list(obj.keys()):
                    obj[k] = harmonize_shapes(obj[k])
                    v = obj.get(k)

                    # Ensure list-shape for selected keys (case-insensitive via keynorm)
                    if keynorm(k) in ALWAYS_LIST_KEYS_NORM:
                        if v is None:
                            obj.pop(k, None)
                            continue
                        if isinstance(v, list):
                            # flatten accidental nesting
                            flat = []
                            for el in v:
                                if isinstance(el, list):
                                    flat.extend(el)
                                else:
                                    flat.append(el)
                            v = flat
                        else:
                            v = [v]

                        # Clean list items
                        cleaned = []
                        for el in v:
                            if el is None:
                                continue
                            if isinstance(el, dict) and not el:
                                continue
                            if isinstance(el, list) and not el:
                                continue
                            # If onLine is given as a plain URL string, wrap it as an object.
                            if keynorm(k) == keynorm('onLine') and isinstance(el, str):
                                cleaned.append({'url': el})
                            else:
                                cleaned.append(el)

                        if cleaned:
                            obj[k] = cleaned
                            v = cleaned
                        else:
                            obj.pop(k, None)
                            continue

                    # Prune empty dict/list placeholders to keep the JSON lean and consistent
                    if isinstance(v, dict) and not v:
                        obj.pop(k, None)
                    elif isinstance(v, list):
                        vv = []
                        for el in v:
                            if el is None:
                                continue
                            if isinstance(el, dict) and not el:
                                continue
                            if isinstance(el, list) and not el:
                                continue
                            vv.append(el)
                        if vv:
                            obj[k] = vv
                        else:
                            obj.pop(k, None)
                return obj

            if isinstance(obj, list):
                out = []
                for el in obj:
                    el = harmonize_shapes(el)
                    if el is None:
                        continue
                    if isinstance(el, dict) and not el:
                        continue
                    if isinstance(el, list) and not el:
                        continue
                    out.append(el)
                return out

            return obj

        self.data = harmonize_shapes(self.data)

        # 9) rename keys
        self.data = rename_key_ci(self.data, 'schedule', 'coverage')

        # 10) final scrub (targeted)
        drop_keys_ci(self.data, {"featureofinterest", "om_observation"})
        # NOTE: do NOT drop None globally (e.g., keep endPosition=None if present)
        # 11) rename(s) last
        if isinstance(self.data, dict) and 'headerInformation' in self.data:
            self.data['header'] = self.data.pop('headerInformation')

        # 11) Unwrap remaining schema-type wrapper:
        #   {"header": {"Header": {...}}} -> {"header": {...}}
        # (This can remain after the headerInformation -> header rename, because
        #  the wrapper name "Header" does not match "headerInformation".)
        if isinstance(self.data, dict):
            hdr = self.data.get("header")
            if isinstance(hdr, dict) and len(hdr) == 1:
                (only_k, only_v), = hdr.items()
                if isinstance(only_v, dict) and same_name(only_k, "header"):
                    self.data["header"] = only_v


    def to_xml(self, output_path: Path | str | None = None, pretty: bool = True, encoding: str = "utf-8") -> str | None:
        """
        Serialize the internal dictionary to XML.

        Args:
            output_path (str | Path, optional): If specified, write to this path. Otherwise return XML as string.
            pretty (bool): Pretty-print the output. Defaults to True.
            encoding (str): Encoding used when writing to file. Defaults to 'utf-8'.

        Returns:
            str | None: XML string if not written to file.
        """
        xml_str = xmltodict.unparse(self.data, pretty=pretty)

        if output_path:
            Path(output_path).write_text(xml_str, encoding=encoding)
            return None
        return xml_str


    def to_json(self) -> str:
        """
        Convert the WMDR metadata to a JSON-formatted string.

        Returns:
            str: JSON string representation of the metadata.
        """
        return json.dumps(self.data, indent=2)


    def to_yaml(self) -> str:
        """
        Convert the WMDR metadata to a YAML-formatted string.

        Returns:
            str: YAML string representation of the metadata.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML export. Install with `pip install pyyaml`.")
        return yaml.dump(self.data, allow_unicode=True, sort_keys=False)


    def extract(self, parts: Union[str, Iterable[str]]) -> Any:
        """
        Return portions of the simplified WMDR10 JSON.

        parts: "header" | "facility" | "observations" | "deployments", or an iterable of them.

        - "header"       -> self.data["header"]
        - "facility"     -> self.data["facility"]
        - "observations" -> self.data["observations"]
        - "deployments"  -> list of deployment stubs, each enriched with:
                            {"facility": <obs.facility>,
                            "observedProperty": <obs.observedProperty>,
                            "type": <obs.type>, ...deployment fields...}

        Notes:
        • No file writing here. Use WMDR10.export(path, fmt, parts=..., index=...) for output.
        • If multiple parts are requested, returns a dict {part: value, ...}.
            If a single part is requested, returns that value directly.
        """
        def _norm(s: str) -> str:
            return str(s).strip().lower()

        def _collect_deployments() -> List[dict]:
            out: List[dict] = []
            obs_list = self.data.get("observations")
            if not isinstance(obs_list, list):
                return out
            for obs in obs_list:
                if not isinstance(obs, dict):
                    continue
                fac = obs.get("facility")
                obs_prop = obs.get("observedProperty")
                typ = obs.get("type")
                deps = obs.get("deployments") or obs.get("deployment")
                if deps is None:
                    continue
                if not isinstance(deps, list):
                    deps = [deps]
                for d in deps:
                    stub = deepcopy(d) if isinstance(d, dict) else {"_deployment": d}
                    stub["facility"] = fac
                    stub["observedProperty"] = obs_prop
                    stub["type"] = typ
                    out.append(stub)
            return out

        if isinstance(parts, str):
            p = _norm(parts)
            if p == "header":
                return self.data.get("header")
            if p == "facility":
                return self.data.get("facility")
            if p == "observations":
                return self.data.get("observations")
            if p == "deployments":
                return _collect_deployments()
            raise ValueError(f"Unknown part: {parts!r}")

        # multiple parts → dict
        result: Dict[str, Any] = {}
        wanted = [_norm(p) for p in parts]
        if "header" in wanted:
            result["header"] = self.data.get("header")
        if "facility" in wanted:
            result["facility"] = self.data.get("facility")
        if "observations" in wanted:
            result["observations"] = self.data.get("observations")
        if "deployments" in wanted:
            result["deployments"] = _collect_deployments()
        return result


    def export(
        self,
        path: Path | str,
        fmt: str = "json",
        *,
        parts: Union[str, Iterable[str], None] = None,
        index: Optional[Union[int, Iterable[int]]] = None,
        minified: bool = False,
    ) -> Path:
        """
        Export this WMDR10 object (or a selected part) to JSON or YAML.

        parts: None (full WMDR10), or one/many of {"header","facility","observations","deployments"}.
        index: If the extracted value is a list, pick an element (0-based). If an iterable of ints is
            provided, apply nested indexing sequentially (e.g. [2, 0] => payload[2][0]).
        """
        # --- Normalize parts for naming and extraction ---
        part_tokens: list[str]
        if parts is None:
            part_tokens = []
        elif isinstance(parts, str):
            part_tokens = [parts]
        else:
            part_tokens = list(parts)

        # Build payload (no index passed to extract)
        payload: Any = self.data if parts is None else self.extract(parts)

        # --- Normalize index (support int or iterable[int]) ---
        index_tokens: list[int] = []
        if index is not None:
            if isinstance(index, numbers.Integral) and not isinstance(index, bool):
                index_tokens = [int(index)]
            else:
                # treat as iterable of ints
                try:
                    index_tokens = [int(i) for i in index]  # type: ignore[arg-type]
                except TypeError as e:
                    raise TypeError("Parameter 'index' must be an int or an iterable of ints.") from e

            # Apply indexing sequentially
            for idx in index_tokens:
                if not isinstance(payload, list):
                    raise TypeError(
                        "Parameter 'index' can only be used when the selected part returns a list "
                        "(e.g., parts='observations' or parts='deployments'), and for nested indexing "
                        "each intermediate value must also be a list."
                    )
                try:
                    payload = payload[idx]
                except IndexError as e:
                    raise IndexError(f"Index {idx} out of range (len={len(payload)}).") from e

        # --- Serialize & write ---
        fmt = fmt.lower()
        ext_map = {"json": ".json", "yaml": ".yaml", "yml": ".yaml"}
        if fmt not in ext_map:
            raise ValueError("fmt must be 'json' or 'yaml'")

        p = Path(path)
        suffix = ext_map[fmt]

        # Build filename: <filename>_<part>_<index_n>_<index_m>_etc.<fmt>
        tokens: list[str] = [p.stem]
        if part_tokens:
            tokens.extend(part_tokens)
        if index_tokens:
            tokens.extend(str(i) for i in index_tokens)

        out_path = p.with_name("_".join(tokens) + suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            if minified:
                txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            else:
                txt = json.dumps(payload, ensure_ascii=False, indent=2)
            out_path.write_text(txt, encoding="utf-8")
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required for YAML export. Install with `pip install pyyaml`.")
            txt = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
            out_path.write_text(txt, encoding="utf-8")

        return out_path


    def _project_uris_from(self, entries: list[Any]) -> list[str]:
        """Extract current programAffiliation URIs from collected entries."""
        # flatten to blocks
        blocks: list[Any] = []
        def push(x):
            if x is None:
                return
            if isinstance(x, list):
                for el in x: push(el)
            elif isinstance(x, dict) and len(x) == 1:
                # unwrap single-key wrapper like {"programAffiliation": [...]}
                (_, v), = x.items()
                push(v)
            else:
                blocks.append(x)
        for e in entries: push(e)

        def is_current(node) -> bool:
            if isinstance(node, str):
                return True
            if not isinstance(node, dict):
                return False
            # ended at block level?
            if node.get('endPosition') not in (None, '', {}):
                return False
            rs = node.get('reportingStatus')
            if rs is None:
                return True
            rs_list = rs if isinstance(rs, list) else [rs]
            for r in rs_list:
                if not isinstance(r, dict):
                    continue
                # direct endPosition or legacy validPeriod.endPosition
                if r.get('endPosition') in (None, '', {}):
                    return True
                vp = r.get('validPeriod')
                if isinstance(vp, dict) and vp.get('endPosition') in (None, '', {}):
                    return True
            return False

        def to_uri(node) -> str | None:
            if isinstance(node, str):
                return node
            if isinstance(node, dict):
                uri = node.get('programAffiliation')
                return uri if isinstance(uri, str) else None
            return None

        seen, uris = set(), []
        for blk in blocks:
            if not is_current(blk):
                continue
            uri = to_uri(blk)
            if uri and uri not in seen:
                seen.add(uri)
                uris.append(uri)
        return uris


    # def _acdd_compute_projects_string(self) -> str:
    #     # Pull all programAffiliation blocks directly from facility
    #     pa = self._resolve_path_recursive(self.data, ["facility", "programAffiliation"])
    #     blocks = pa if isinstance(pa, list) else ([pa] if pa is not None else [])

    #     def is_current(node) -> bool:
    #         if isinstance(node, str):
    #             return True
    #         if not isinstance(node, dict):
    #             return False
    #         # ended?
    #         if node.get('endPosition') not in (None, '', {}):
    #             return False
    #         rs = node.get('reportingStatus')
    #         if rs is None:
    #             return True
    #         rs_list = rs if isinstance(rs, list) else [rs]
    #         for r in rs_list:
    #             if isinstance(r, dict):
    #                 vp = r.get('validPeriod')
    #                 if isinstance(vp, dict) and vp.get('endPosition') in (None, '', {}):
    #                     return True
    #         return False

    #     def to_uri(node) -> str | None:
    #         if isinstance(node, str):
    #             return node
    #         if isinstance(node, dict) and isinstance(node.get('programAffiliation'), str):
    #             return node['programAffiliation']
    #         return None

    #     seen = set()
    #     uris: list[str] = []
    #     for blk in blocks:
    #         if not is_current(blk):
    #             continue
    #         uri = to_uri(blk)
    #         if uri and uri not in seen:
    #             seen.add(uri)
    #             uris.append(uri)

    #     return ", ".join(uris) if uris else ""


    # def _acdd_keywords_as_jsonarray_string(self, entries: list[Any]) -> str:
    #     """
    #     Turn the collected 'keywords' row outputs into a JSON-array string.
    #     Each mapping-row result stays as the JSON we produced (dict/list/scalar).
    #     We de-duplicate while preserving order, using structural equality.
    #     """
    #     def _norm(x: Any) -> Any:
    #         # Keep native dict/list/str/etc.; do NOT stringify here
    #         return x

    #     # de-dupe by structural JSON representation
    #     seen: set[str] = set()
    #     uniq: list[Any] = []
    #     for e in entries:
    #         n = _norm(e)
    #         sig = json.dumps(n, sort_keys=True, separators=(",", ":"))
    #         if sig not in seen:
    #             seen.add(sig)
    #             uniq.append(n)

    #     # return as a JSON string of an array
    #     return json.dumps(uniq, sort_keys=True, separators=(",", ":"))


    # def _acdd_keywords_as_string(self, entries: list[Any]) -> str:
    #     def to_blob(x) -> str:
    #         # Keep JSON structure: dict/list → compact JSON; scalars → str()
    #         if isinstance(x, (dict, list)):
    #             return json.dumps(x, separators=(",", ":"), sort_keys=True)
    #         return str(x)

    #     # Each mapping row may have returned a wrapper like {<key>: value}
    #     blobs: list[str] = []
    #     for e in entries:
    #         if isinstance(e, dict) and len(e) == 1:
    #             # keep the one-key wrapper as-is (it's the JSON you expect)
    #             blobs.append(to_blob(e))
    #         else:
    #             # also accept plain dicts/lists/scalars
    #             blobs.append(to_blob(e))

    #     # dedupe while preserving order
    #     seen = set()
    #     uniq = []
    #     for b in blobs:
    #         if b not in seen:
    #             seen.add(b)
    #             uniq.append(b)

    #     return ", ".join(uniq)


    def _create_json_stub(self, raw: dict, row: dict) -> dict | list | str | None:
        """
        Return native values resolved from 'wmdr10_simplified_path'.
        If 'keywords_key' is present, wrap as {<key>: value} (preserve structure).
        """
        path = (row.get("wmdr10_simplified_path") or "").split("/")
        key  = row.get("keywords_key") or row.get("acdd_keywords_key")

        value = self._resolve_path_recursive(deepcopy(raw), path)
        if value is None:
            return None
        return {key: value} if key else value


    # def _create_geospatial_attributes(self, mapping_row: dict[str, str], delta: float=0.001) -> dict[str, float | str]:
    #     """
    #     Extract geolocation bounding box info from pos string in the mapping and convert to ACDD fields.

    #     Args:
    #         mapping_rows (list[dict[str, str]]): List of mapping rows.
    #         delta(float, optional): Slack applied to generate geospatial bounding box from single geolocation

    #     Returns:
    #         dict[str, Any]: Geospatial ACDD attributes.
    #     """
    #     def geospatial_bounds(lat, lon, d):
    #         return f"POLYGON(({lat-d} {lon-d}, {lat-d} {lon+d}, {lat+d} {lon+d}, {lat+d} {lon-d}, {lat-d} {lon-d}))"
        
    #     result = dict()

    #     required_keys = ["acdd_attribute", "wmdr10_simplified_path", "default"]

    #     if not all(k in mapping_row for k in required_keys):
    #         # [TODO] issue warning
    #         warnings.warn("_create_geospatial_attributes: some required_keys are missing.")
    #         return {}  # skip incomplete rows

    #     if not mapping_row["acdd_attribute"].startswith("geospatial_"):
    #         return {}

    #     # generate geospatial_ elements
    #     if mapping_row["wmdr10_simplified_path"].endswith("geospatialLocation"):
    #         path = mapping_row["wmdr10_simplified_path"].split("/")
    #         val = self._resolve_path_recursive(self.data, path)
    #         if isinstance(val, list):                
    #             val = max(val,
    #                       key=lambda x: datetime.fromisoformat(x.get('beginPosition', '0001-01-01T00:00:00+00:00').replace('Z', '+00:00'))
    #                       )['geoLocation']

    #         if isinstance(val, dict):
    #             val = val['geoLocation']

    #         if isinstance(val, str):
    #             try:
    #                 lat_str, lon_str, *alt_str = val.strip().split()
    #                 lat, lon = float(lat_str), float(lon_str)
    #                 alt = float(alt_str[0]) if alt_str else None
    #                 result = {
    #                     "geospatial_lat_min": lat - delta,
    #                     "geospatial_lat_max": lat + delta,
    #                     "geospatial_lon_min": lon - delta,
    #                     "geospatial_lon_max": lon + delta,
    #                     "geospatial_bounds": geospatial_bounds(lat, lon, delta),
    #                     "geospatial_bounds_crs": "WGS84",
    #                     "geospatial_bounds_vertical_crs": "EPSG:5829",
    #                     "geospatial_lat_units": "degree_north",
    #                     "geospatial_lon_units": "degree_east",
    #                     "geospatial_vertical_positive": "up",
    #                 }
    #                 if alt is not None:
    #                     result.update({
    #                         "geospatial_vertical_min": alt - 5000 * delta,
    #                         "geospatial_vertical_max": alt + 5000 * delta,
    #                         "geospatial_bounds_vertical_crs": "EPSG:5829",
    #                         "geospatial_vertical_units": "EPSG:4979",
    #                     })
    #                 return result
    #             except ValueError:
    #                 return {}
    #     return {}


    def _resolve_path_recursive(self, obj: Any, path_parts: Sequence[str]) -> Any:
        parts = list(path_parts)
        if not parts:
            return obj

        head, *tail = parts
        wanted = self._normalize_key(head)

        if isinstance(obj, list):
            results = []
            for item in obj:
                r = self._resolve_path_recursive(item, parts)
                if r is None:
                    continue
                if isinstance(r, list):
                    results.extend(r)
                else:
                    results.append(r)
            return results if results else None

        if isinstance(obj, dict):
            # 1) exact key hit first
            if head in obj:
                return self._resolve_path_recursive(obj[head], tail)
            # 2) normalized equality (safer than startswith/endswith)
            for k, v in obj.items():
                if self._normalize_key(k) == wanted:
                    return self._resolve_path_recursive(v, tail)
            return None

        return None


    # def observation_to_acdd(self, mapping_file: Path | str) -> ACDD:
    #     raise NotImplementedError


# What the campaign machine has to be allowed to do

A campaign VM does not run as the person who launched it. It runs as the
project's **default compute service account**,
`<project-number>-compute@developer.gserviceaccount.com`, and that account's
permissions are separate from the operator's. On a project where the default
account carries no project-level role, a machine boots, passes every gate,
computes for hours, and writes nothing anywhere.

That is not hypothetical. The first `frozen_n16_catchup` launch cleared the
quota preflight, the pinned-commit check, the twelve self-tests and the mini
dry run, then failed every single sync pass with

    HTTPError 403: ...-compute@developer.gserviceaccount.com does not have
    storage.objects.get access to the ... object

into a serial console nobody was reading. `cloud/provision.sh` now checks both
grants below before it creates anything.

## The two grants

**Results.** Without this the run produces nothing that outlives the instance,
and it fails silently: the campaign keeps computing, the sync loop keeps
retrying, and a spot machine that is preempted takes every archive with it.

```bash
gcloud storage buckets add-iam-policy-binding gs://<project-id>-pql-campaign --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/storage.objectAdmin
```

**Teardown.** The finisher's last act is to resize its own managed instance
group to zero. Without this it reports `TEARDOWN=denied`, writes an
`ACTION_REQUIRED_*` object, and leaves a 96-vCPU spot machine billing — after a
run that *succeeded*. This is the grant that costs money when it is missing.

```bash
gcloud projects add-iam-policy-binding <project-id> --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/compute.instanceAdmin.v1
```

## Scopes are not permissions

The instance template requests `--scopes=cloud-platform`. A scope is a ceiling
on what a token may be used for, not a grant: `cloud-platform` with no IAM role
still authorizes nothing. The template previously requested `storage-rw`, which
capped the machine below what its own finisher needed — the teardown call was
refused by the *scope* before IAM was ever consulted. Both have to be right,
and they fail with different-looking errors.

## Checking by hand

```bash
gcloud projects get-iam-policy <project-id> --flatten='bindings[]' --format='value(bindings.role)' --filter='bindings.members:serviceAccount:<project-number>-compute@developer.gserviceaccount.com'
```

An empty result means the machine can do nothing outside its own disk.

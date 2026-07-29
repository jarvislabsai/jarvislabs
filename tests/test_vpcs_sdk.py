"""SDK tests for the VPC resource and vpc_id on VM create/resume."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from jarvislabs.constants import CHENNAI_REGION, INDIA_NOIDA_REGION, REGION_URLS
from jarvislabs.exceptions import APIError, NotFoundError, ValidationError
from jarvislabs.instances import Instances
from jarvislabs.models import SSHKey
from jarvislabs.vpcs import Vpcs

_DUMMY_KEY = SSHKey(ssh_key="ssh-ed25519 AAA", key_name="test", key_id="k1")

_VPC_CHENNAI = {
    "vpc_id": "vpc-aaaa",
    "name": "training-net",
    "region": CHENNAI_REGION,
    "cidr": "10.50.0.0/24",
    "gateway_ip": "10.50.0.1",
    "is_default": False,
    "status": "active",
}

_VPC_DEFAULT_NOIDA = {
    "vpc_id": "vpc-bbbb",
    "name": "IN2 DEFAULT VPC",
    "region": INDIA_NOIDA_REGION,
    "cidr": "10.0.0.0/16",
    "gateway_ip": "10.0.0.1",
    "is_default": True,
    "status": "active",
}


def _make_instances(mock_transport):
    ssh_keys = MagicMock(**{"list.return_value": [_DUMMY_KEY]})
    return Instances(mock_transport, ssh_keys)


def _paused_vm(**overrides):
    defaults = {
        "machine_id": 10,
        "template": "vm",
        "gpu_type": "RTX5000",
        "num_gpus": 1,
        "storage_gb": 40,
        "fs_id": None,
        "region": INDIA_NOIDA_REGION,
        "status": "Paused",
    }
    defaults.update(overrides)
    m = MagicMock(**defaults)
    m.configure_mock(name="old-name")
    return m


# ── Vpcs.list / get ──────────────────────────────────────────────────────────


class TestVpcListGet:
    def test_list_parses_and_uses_default_backend(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI, _VPC_DEFAULT_NOIDA]

        vpcs = Vpcs(mock_transport).list()

        mock_transport.request.assert_called_once_with("GET", "vpc/list")
        assert [v.vpc_id for v in vpcs] == ["vpc-aaaa", "vpc-bbbb"]
        assert vpcs[0].region == CHENNAI_REGION
        assert vpcs[1].is_default is True

    def test_list_rejects_unexpected_response(self, mock_transport):
        mock_transport.request.return_value = {"message": "nope"}

        with pytest.raises(APIError, match="unexpected response"):
            Vpcs(mock_transport).list()

    def test_get_returns_matching_vpc(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI, _VPC_DEFAULT_NOIDA]

        vpc = Vpcs(mock_transport).get("vpc-bbbb")

        assert vpc.name == "IN2 DEFAULT VPC"

    def test_get_raises_not_found(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(NotFoundError, match="vpc-missing"):
            Vpcs(mock_transport).get("vpc-missing")

    def test_region_serializes_to_display_code(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        dumped = Vpcs(mock_transport).list()[0].model_dump()

        assert dumped["region"] == "IN1"


# ── Vpcs.create ──────────────────────────────────────────────────────────────


class TestVpcCreate:
    def test_create_routes_to_region_and_sends_payload(self, mock_transport):
        mock_transport.request.return_value = _VPC_CHENNAI

        vpc = Vpcs(mock_transport).create(name="training-net", cidr="10.50.0.0/24", region="IN1")

        mock_transport.request.assert_called_once_with(
            "POST",
            "vpc/create",
            json={"name": "training-net", "cidr": "10.50.0.0/24", "region": CHENNAI_REGION},
            base_url=REGION_URLS[CHENNAI_REGION],
        )
        assert vpc.vpc_id == "vpc-aaaa"

    def test_create_accepts_internal_region_id(self, mock_transport):
        mock_transport.request.return_value = _VPC_CHENNAI

        Vpcs(mock_transport).create(name="n", cidr="10.50.0.0/24", region=CHENNAI_REGION)

        assert mock_transport.request.call_args.kwargs["json"]["region"] == CHENNAI_REGION

    def test_create_rejects_unknown_region_before_any_request(self, mock_transport):
        with pytest.raises(ValidationError, match="Unknown region"):
            Vpcs(mock_transport).create(name="n", cidr="10.50.0.0/24", region="mars-01")

        mock_transport.request.assert_not_called()

    def test_create_requires_region(self, mock_transport):
        with pytest.raises(ValidationError, match="requires a region"):
            Vpcs(mock_transport).create(name="n", cidr="10.50.0.0/24", region="  ")

        mock_transport.request.assert_not_called()

    def test_create_rejects_unexpected_response(self, mock_transport):
        mock_transport.request.return_value = {"message": "weird"}

        with pytest.raises(APIError, match="weird"):
            Vpcs(mock_transport).create(name="n", cidr="10.50.0.0/24", region="IN1")


# ── Vpcs.ips ─────────────────────────────────────────────────────────────────


class TestVpcIps:
    def test_ips_resolves_region_then_fetches(self, mock_transport):
        mock_transport.request.side_effect = [
            [_VPC_CHENNAI],
            [
                {
                    "lsp_name": "vpc-aaaa-10-50-0-2",
                    "private_ip": "10.50.0.2",
                    "mac_address": "42:a1:b2:c3:d4:e5",
                    "machine_id": 84512,
                    "status": "attached",
                },
                {
                    "lsp_name": "vpc-aaaa-10-50-0-3",
                    "private_ip": "10.50.0.3",
                    "mac_address": "42:f6:07:18:29:3a",
                    "machine_id": None,
                    "status": "reserved",
                },
            ],
        ]

        ips = Vpcs(mock_transport).ips("vpc-aaaa")

        assert mock_transport.request.call_args_list == [
            call("GET", "vpc/list"),
            call("GET", "vpc/vpc-aaaa/ports", base_url=REGION_URLS[CHENNAI_REGION]),
        ]
        assert [(ip.private_ip, ip.machine_id, ip.status) for ip in ips] == [
            ("10.50.0.2", 84512, "attached"),
            ("10.50.0.3", None, "reserved"),
        ]
        assert ips[0].lsp_name == "vpc-aaaa-10-50-0-2"

    def test_ips_unknown_vpc_raises_before_ports_call(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(NotFoundError):
            Vpcs(mock_transport).ips("vpc-missing")

        mock_transport.request.assert_called_once_with("GET", "vpc/list")

    def test_ips_rejects_unexpected_response(self, mock_transport):
        mock_transport.request.side_effect = [[_VPC_CHENNAI], {"message": "nope"}]

        with pytest.raises(APIError, match="unexpected response"):
            Vpcs(mock_transport).ips("vpc-aaaa")


# ── Vpcs.delete ──────────────────────────────────────────────────────────────


class TestVpcDelete:
    def test_delete_routes_to_region(self, mock_transport):
        mock_transport.request.side_effect = [[_VPC_CHENNAI], {"success": True}]

        assert Vpcs(mock_transport).delete("vpc-aaaa") is True

        assert mock_transport.request.call_args_list == [
            call("GET", "vpc/list"),
            call("DELETE", "vpc/vpc-aaaa", base_url=REGION_URLS[CHENNAI_REGION]),
        ]

    def test_delete_unknown_vpc_raises_before_delete_call(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(NotFoundError):
            Vpcs(mock_transport).delete("vpc-missing")

        mock_transport.request.assert_called_once_with("GET", "vpc/list")

    def test_delete_raises_on_declared_failure(self, mock_transport):
        mock_transport.request.side_effect = [[_VPC_CHENNAI], {"success": False, "message": "still attached"}]

        with pytest.raises(APIError, match="still attached"):
            Vpcs(mock_transport).delete("vpc-aaaa")


# ── vpc_id on instance create ────────────────────────────────────────────────


class TestInstanceCreateVpc:
    @patch("jarvislabs.instances._get_instance")
    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._validate_create_region")
    def test_vm_create_sends_vpc_id(self, _region, _poll, mock_get, mock_transport):
        mock_transport.request.side_effect = [[_VPC_DEFAULT_NOIDA], {"machine_id": 1}]
        mock_get.return_value = MagicMock(machine_id=1)

        _make_instances(mock_transport).create(gpu_type="RTX5000", template="vm", region="IN2", vpc_id="vpc-bbbb")

        assert mock_transport.request.call_args.kwargs["json"]["vpc_id"] == "vpc-bbbb"

    @patch("jarvislabs.instances._get_instance")
    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._validate_create_region")
    @patch("jarvislabs.instances._resolve_region")
    def test_vm_create_infers_region_from_vpc(self, mock_resolve, _region, _poll, mock_get, mock_transport):
        mock_transport.request.side_effect = [[_VPC_CHENNAI], {"machine_id": 1}]
        mock_get.return_value = MagicMock(machine_id=1)

        _make_instances(mock_transport).create(gpu_type="RTX5000", template="vm", vpc_id="vpc-aaaa")

        mock_resolve.assert_not_called()
        assert mock_transport.request.call_args.kwargs["json"]["region"] == CHENNAI_REGION
        assert mock_transport.request.call_args.kwargs["base_url"] == REGION_URLS[CHENNAI_REGION]

    def test_vm_create_rejects_vpc_region_mismatch(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(ValidationError, match="same region"):
            _make_instances(mock_transport).create(gpu_type="RTX5000", template="vm", region="IN2", vpc_id="vpc-aaaa")

        mock_transport.request.assert_called_once_with("GET", "vpc/list")

    def test_vm_create_rejects_unknown_vpc(self, mock_transport):
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(NotFoundError, match="vpc-missing not found"):
            _make_instances(mock_transport).create(gpu_type="RTX5000", template="vm", vpc_id="vpc-missing")

        mock_transport.request.assert_called_once_with("GET", "vpc/list")

    @patch("jarvislabs.instances._get_instance")
    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._validate_create_region")
    def test_vm_create_omits_vpc_id_when_unset(self, _region, _poll, mock_get, mock_transport):
        mock_transport.request.return_value = {"machine_id": 1}
        mock_get.return_value = MagicMock(machine_id=1)

        _make_instances(mock_transport).create(gpu_type="RTX5000", template="vm", region="IN2")

        assert "vpc_id" not in mock_transport.request.call_args.kwargs["json"]

    def test_container_create_rejects_vpc_id(self, mock_transport):
        with pytest.raises(ValidationError, match="only supported for VM"):
            _make_instances(mock_transport).create(gpu_type="L4", template="pytorch", vpc_id="vpc-aaaa")

        mock_transport.request.assert_not_called()

    @patch("jarvislabs.instances._get_instance")
    @patch("jarvislabs.instances._poll_until_running")
    def test_cpu_vm_create_sends_vpc_id(self, _poll, mock_get, mock_transport):
        mock_transport.request.side_effect = [
            [_VPC_DEFAULT_NOIDA],
            {
                "server_meta": [],
                "cpu_meta": {
                    "combinations": [
                        {
                            "vcpus": 4,
                            "ram_gb": 16,
                            "price": 0.12,
                            "available": True,
                            "regions": {INDIA_NOIDA_REGION: True},
                        }
                    ]
                },
            },
            {"machine_id": 1},
        ]
        mock_get.return_value = MagicMock(machine_id=1)

        _make_instances(mock_transport).create(template="vm", cpu=True, storage=100, vpc_id="vpc-bbbb")

        assert mock_transport.request.call_args.args == ("POST", "templates/vm/cpu/create")
        assert mock_transport.request.call_args.kwargs["json"]["vpc_id"] == "vpc-bbbb"


# ── vpc_id on instance resume ────────────────────────────────────────────────


class TestInstanceResumeVpc:
    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._get_instance")
    def test_vm_resume_sends_vpc_id(self, mock_get, _poll, mock_transport):
        mock_get.return_value = _paused_vm()
        mock_transport.request.side_effect = [[_VPC_DEFAULT_NOIDA], {"machine_id": 11}]

        _make_instances(mock_transport).resume(10, vpc_id="vpc-bbbb")

        assert mock_transport.request.call_args.args == ("POST", "templates/vm/resume")
        assert mock_transport.request.call_args.kwargs["json"]["vpc_id"] == "vpc-bbbb"

    @patch("jarvislabs.instances._get_instance")
    def test_vm_resume_rejects_vpc_region_mismatch(self, mock_get, mock_transport):
        mock_get.return_value = _paused_vm()
        mock_transport.request.return_value = [_VPC_CHENNAI]

        with pytest.raises(ValidationError, match="same region"):
            _make_instances(mock_transport).resume(10, vpc_id="vpc-aaaa")

        mock_transport.request.assert_called_once_with("GET", "vpc/list")

    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._get_instance")
    def test_vm_resume_omits_vpc_id_to_keep_current_vpc(self, mock_get, _poll, mock_transport):
        mock_get.return_value = _paused_vm()
        mock_transport.request.return_value = {"machine_id": 11}

        _make_instances(mock_transport).resume(10)

        assert "vpc_id" not in mock_transport.request.call_args.kwargs["json"]

    @patch("jarvislabs.instances._get_instance")
    def test_container_resume_rejects_vpc_id(self, mock_get, mock_transport):
        mock_get.return_value = _paused_vm(template="pytorch")

        with pytest.raises(ValidationError, match="only supported for VM"):
            _make_instances(mock_transport).resume(10, vpc_id="vpc-aaaa")

        mock_transport.request.assert_not_called()

    @patch("jarvislabs.instances._poll_until_running")
    @patch("jarvislabs.instances._get_instance")
    def test_cpu_vm_resume_sends_vpc_id(self, mock_get, _poll, mock_transport):
        mock_get.return_value = _paused_vm(gpu_type="CPU")
        mock_transport.request.side_effect = [[_VPC_DEFAULT_NOIDA], {"machine_id": 11}]

        _make_instances(mock_transport).resume(10, vpc_id="vpc-bbbb")

        assert mock_transport.request.call_args.args == ("POST", "templates/vm/cpu/resume")
        assert mock_transport.request.call_args.kwargs["json"]["vpc_id"] == "vpc-bbbb"
